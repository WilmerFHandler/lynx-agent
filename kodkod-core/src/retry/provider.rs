use crate::{AssistantMessage, Conversation, Provider, ToolSpec};

use super::{RetryPolicy, Retryable};

/// Wraps any [`Provider`] whose [`Provider::Error`] implements [`Retryable`].
#[derive(Debug, Clone)]
pub struct RetryProvider<P> {
    inner: P,
    policy: RetryPolicy,
}

impl<P> RetryProvider<P> {
    pub fn new(inner: P) -> Self {
        Self {
            inner,
            policy: RetryPolicy::default(),
        }
    }

    pub fn with_policy(inner: P, policy: RetryPolicy) -> Self {
        Self { inner, policy }
    }

    pub fn inner(&self) -> &P {
        &self.inner
    }

    pub fn into_inner(self) -> P {
        self.inner
    }

    pub fn policy(&self) -> &RetryPolicy {
        &self.policy
    }
}

impl<P> Provider for RetryProvider<P>
where
    P: Provider + Sync,
    P::Error: Retryable,
    P::TurnState: Clone,
{
    type Model = P::Model;
    type Error = P::Error;
    type TurnState = P::TurnState;

    fn create_turn_state(&self, model: &Self::Model) -> Self::TurnState {
        self.inner.create_turn_state(model)
    }

    fn supports_vision(&self, model: &Self::Model) -> bool {
        self.inner.supports_vision(model)
    }

    fn supports_computer_use(&self, model: &Self::Model) -> bool {
        self.inner.supports_computer_use(model)
    }

    async fn complete_round(
        &self,
        state: &mut Self::TurnState,
        model: &Self::Model,
        conversation: &Conversation,
        tools: &[ToolSpec],
    ) -> Result<AssistantMessage, Self::Error> {
        let max = self.policy.max_attempts.max(1);
        let mut attempt = 0u32;

        loop {
            attempt += 1;
            // Clone must be a logically independent snapshot. Shared interior
            // mutability cannot provide rollback of local continuation state.
            let mut attempt_state = state.clone();
            match self
                .inner
                .complete_round(&mut attempt_state, model, conversation, tools)
                .await
            {
                Ok(message) => {
                    *state = attempt_state;
                    return Ok(message);
                }
                Err(error) if error.is_retryable() && attempt < max => {
                    futures_timer::Delay::new(self.policy.backoff_after_attempt(attempt)).await;
                }
                Err(error) => return Err(error),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;
    use std::fmt;
    use std::future::Future;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Duration;

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct RetryTestError {
        retryable: bool,
        status_code: Option<u16>,
    }

    impl RetryTestError {
        fn http(status_code: u16, retryable: bool) -> Self {
            Self {
                retryable,
                status_code: Some(status_code),
            }
        }
    }

    impl fmt::Display for RetryTestError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "retry test error")
        }
    }

    impl Error for RetryTestError {}

    impl Retryable for RetryTestError {
        fn is_retryable(&self) -> bool {
            self.retryable
        }
    }

    #[derive(Clone, Debug)]
    struct TestModel;

    impl TestModel {
        fn vision(&self) -> bool {
            false
        }
    }

    struct FlakyProvider {
        calls: Arc<AtomicU32>,
        fail_until: u32,
    }

    impl Provider for FlakyProvider {
        type Model = TestModel;
        type Error = RetryTestError;
        type TurnState = ();

        fn create_turn_state(&self, _model: &Self::Model) -> Self::TurnState {}

        fn supports_vision(&self, model: &TestModel) -> bool {
            model.vision()
        }

        fn complete_round(
            &self,
            _state: &mut Self::TurnState,
            _model: &TestModel,
            _conversation: &Conversation,
            _tools: &[ToolSpec],
        ) -> impl Future<Output = Result<AssistantMessage, RetryTestError>> + Send {
            let calls = Arc::clone(&self.calls);
            let fail_until = self.fail_until;
            async move {
                let n = calls.fetch_add(1, Ordering::SeqCst) + 1;
                if n <= fail_until {
                    Err(RetryTestError::http(503, true))
                } else {
                    Ok(AssistantMessage::new("ok"))
                }
            }
        }
    }

    #[tokio::test]
    async fn retries_retryable_http_until_success() {
        let inner = FlakyProvider {
            calls: Arc::new(AtomicU32::new(0)),
            fail_until: 2,
        };
        let provider = RetryProvider::with_policy(
            inner,
            RetryPolicy {
                max_attempts: 4,
                initial_backoff: Duration::from_millis(1),
                max_backoff: Duration::from_millis(5),
                backoff_multiplier: 2.0,
            },
        );

        let model = TestModel;
        let conversation = Conversation::new();
        let message = provider
            .complete_once(&model, &conversation, &[])
            .await
            .expect("should succeed after retries");

        assert_eq!(message.content(), "ok");
    }

    #[tokio::test]
    async fn does_not_retry_non_retryable_errors() {
        struct Once401;
        impl Provider for Once401 {
            type Model = TestModel;
            type Error = RetryTestError;
            type TurnState = ();

            fn create_turn_state(&self, _model: &Self::Model) -> Self::TurnState {}

            fn supports_vision(&self, model: &TestModel) -> bool {
                model.vision()
            }

            async fn complete_round(
                &self,
                _state: &mut Self::TurnState,
                _model: &TestModel,
                _conversation: &Conversation,
                _tools: &[ToolSpec],
            ) -> Result<AssistantMessage, RetryTestError> {
                Err(RetryTestError::http(401, false))
            }
        }

        let model = TestModel;
        let provider = RetryProvider::new(Once401);
        let err = provider
            .complete_once(&model, &Conversation::new(), &[])
            .await
            .unwrap_err();

        assert_eq!(err.status_code, Some(401));
    }

    #[tokio::test]
    async fn dropping_completion_during_backoff_stops_retries() {
        let calls = Arc::new(AtomicU32::new(0));
        let provider = RetryProvider::with_policy(
            FlakyProvider {
                calls: Arc::clone(&calls),
                fail_until: u32::MAX,
            },
            RetryPolicy {
                max_attempts: 4,
                initial_backoff: Duration::from_millis(20),
                max_backoff: Duration::from_millis(20),
                backoff_multiplier: 1.0,
            },
        );
        let model = TestModel;
        let conversation = Conversation::new();
        let mut completion = Box::pin(provider.complete_once(&model, &conversation, &[]));

        tokio::time::timeout(Duration::from_millis(5), completion.as_mut())
            .await
            .expect_err("retry should still be in backoff");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        drop(completion);

        tokio::time::sleep(Duration::from_millis(40)).await;
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}

#[cfg(test)]
mod transactional_tests {
    use super::*;
    use std::error::Error;
    use std::fmt;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[derive(Debug)]
    struct E;
    impl fmt::Display for E {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("retry")
        }
    }
    impl Error for E {}
    impl Retryable for E {
        fn is_retryable(&self) -> bool {
            true
        }
    }

    struct MutatingProvider(Arc<AtomicUsize>);
    impl Provider for MutatingProvider {
        type Model = ();
        type Error = E;
        type TurnState = usize;
        fn supports_vision(&self, _: &()) -> bool {
            false
        }
        fn create_turn_state(&self, _: &()) -> usize {
            0
        }
        async fn complete_round(
            &self,
            state: &mut usize,
            _: &(),
            _: &Conversation,
            _: &[ToolSpec],
        ) -> Result<AssistantMessage, E> {
            *state += 1;
            let call = self.0.fetch_add(1, Ordering::SeqCst);
            if call < 2 {
                Err(E)
            } else {
                Ok(AssistantMessage::new("ok"))
            }
        }
    }

    #[tokio::test]
    async fn failed_attempt_state_is_rolled_back_and_success_is_committed() {
        let provider = RetryProvider::with_policy(
            MutatingProvider(Arc::new(AtomicUsize::new(0))),
            RetryPolicy {
                max_attempts: 3,
                initial_backoff: Duration::ZERO,
                max_backoff: Duration::ZERO,
                backoff_multiplier: 1.0,
            },
        );
        let mut state = provider.create_turn_state(&());
        provider
            .complete_round(&mut state, &(), &Conversation::new(), &[])
            .await
            .unwrap();
        assert_eq!(
            state, 1,
            "only the successful attempt snapshot is committed"
        );
    }
}
