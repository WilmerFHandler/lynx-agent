use futures::StreamExt;

use crate::{AssistantMessage, Conversation, Provider, ProviderEvent, ProviderStream, ToolSpec};

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
{
    type Model = P::Model;
    type Error = P::Error;
    type Continuation = P::Continuation;

    fn create_continuation(&self, model: &Self::Model) -> Self::Continuation {
        self.inner.create_continuation(model)
    }

    fn supports_vision(&self, model: &Self::Model) -> bool {
        self.inner.supports_vision(model)
    }

    fn supports_document(&self, model: &Self::Model, mime: &str) -> bool {
        self.inner.supports_document(model, mime)
    }

    fn supports_computer_use(&self, model: &Self::Model) -> bool {
        self.inner.supports_computer_use(model)
    }

    async fn complete(
        &self,
        continuation: &Self::Continuation,
        model: &Self::Model,
        conversation: &Conversation,
        tools: &[ToolSpec],
    ) -> Result<(AssistantMessage, Self::Continuation), Self::Error> {
        let max = self.policy.max_attempts.max(1);
        let mut attempt = 0u32;

        loop {
            attempt += 1;
            match self
                .inner
                .complete(continuation, model, conversation, tools)
                .await
            {
                Ok(completion) => return Ok(completion),
                Err(error) if error.is_retryable() && attempt < max => {
                    futures_timer::Delay::new(self.policy.backoff_after_attempt(attempt)).await;
                }
                Err(error) => return Err(error),
            }
        }
    }

    fn complete_stream<'a>(
        &'a self,
        continuation: &'a Self::Continuation,
        model: &'a Self::Model,
        conversation: &'a Conversation,
        tools: &'a [ToolSpec],
    ) -> ProviderStream<'a, Self::Continuation, Self::Error> {
        Box::pin(async_stream::try_stream! {
            let max = self.policy.max_attempts.max(1);
            let mut attempt = 0u32;
            loop {
                attempt += 1;
                let mut stream = self.inner.complete_stream(
                    continuation,
                    model,
                    conversation,
                    tools,
                );
                let mut emitted_text = false;
                let mut retry = false;
                while let Some(event) = stream.next().await {
                    match event {
                        Ok(ProviderEvent::TextDelta(delta)) => {
                            emitted_text = true;
                            yield ProviderEvent::TextDelta(delta);
                        }
                        Ok(ProviderEvent::Completed(message, continuation)) => {
                            yield ProviderEvent::Completed(message, continuation);
                            return;
                        }
                        Err(error)
                            if !emitted_text && error.is_retryable() && attempt < max =>
                        {
                            retry = true;
                            break;
                        }
                        Err(error) => Err(error)?,
                    }
                }
                drop(stream);
                if !retry {
                    return;
                }
                futures_timer::Delay::new(self.policy.backoff_after_attempt(attempt)).await;
            }
        })
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
        type Continuation = ();

        fn create_continuation(&self, _model: &Self::Model) -> Self::Continuation {}

        fn supports_vision(&self, model: &TestModel) -> bool {
            model.vision()
        }

        fn complete(
            &self,
            _state: &Self::Continuation,
            _model: &TestModel,
            _conversation: &Conversation,
            _tools: &[ToolSpec],
        ) -> impl Future<Output = Result<(AssistantMessage, Self::Continuation), RetryTestError>> + Send
        {
            let calls = Arc::clone(&self.calls);
            let fail_until = self.fail_until;
            async move {
                let n = calls.fetch_add(1, Ordering::SeqCst) + 1;
                if n <= fail_until {
                    Err(RetryTestError::http(503, true))
                } else {
                    Ok((AssistantMessage::new("ok"), ()))
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
            type Continuation = ();

            fn create_continuation(&self, _model: &Self::Model) -> Self::Continuation {}

            fn supports_vision(&self, model: &TestModel) -> bool {
                model.vision()
            }

            async fn complete(
                &self,
                _state: &Self::Continuation,
                _model: &TestModel,
                _conversation: &Conversation,
                _tools: &[ToolSpec],
            ) -> Result<(AssistantMessage, Self::Continuation), RetryTestError> {
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

    struct StreamingFailure {
        calls: Arc<AtomicU32>,
        fail_after_delta: bool,
    }

    impl Provider for StreamingFailure {
        type Model = TestModel;
        type Error = RetryTestError;
        type Continuation = ();

        fn create_continuation(&self, _: &TestModel) {}
        fn supports_vision(&self, _: &TestModel) -> bool {
            false
        }

        async fn complete(
            &self,
            _: &(),
            _: &TestModel,
            _: &Conversation,
            _: &[ToolSpec],
        ) -> Result<(AssistantMessage, ()), RetryTestError> {
            unreachable!()
        }

        fn complete_stream<'a>(
            &'a self,
            _: &'a (),
            _: &'a TestModel,
            _: &'a Conversation,
            _: &'a [ToolSpec],
        ) -> ProviderStream<'a, (), RetryTestError> {
            let attempt = self.calls.fetch_add(1, Ordering::SeqCst);
            let fail_after_delta = self.fail_after_delta;
            Box::pin(async_stream::try_stream! {
                if fail_after_delta {
                    yield ProviderEvent::TextDelta("once".into());
                    Err(RetryTestError::http(503, true))?;
                } else if attempt == 0 {
                    Err(RetryTestError::http(503, true))?;
                } else {
                    yield ProviderEvent::Completed(AssistantMessage::new("ok"), ());
                }
            })
        }
    }

    #[tokio::test]
    async fn streaming_retries_only_before_any_text_is_emitted() {
        let policy = RetryPolicy {
            max_attempts: 2,
            initial_backoff: Duration::ZERO,
            max_backoff: Duration::ZERO,
            backoff_multiplier: 1.0,
        };
        let calls = Arc::new(AtomicU32::new(0));
        let provider = RetryProvider::with_policy(
            StreamingFailure {
                calls: Arc::clone(&calls),
                fail_after_delta: false,
            },
            policy.clone(),
        );
        let conversation = Conversation::new();
        let mut stream = provider.complete_stream(&(), &TestModel, &conversation, &[]);
        assert!(
            matches!(stream.next().await.unwrap().unwrap(), ProviderEvent::Completed(message, ()) if message.content() == "ok")
        );
        assert_eq!(calls.load(Ordering::SeqCst), 2);

        let calls = Arc::new(AtomicU32::new(0));
        let provider = RetryProvider::with_policy(
            StreamingFailure {
                calls: Arc::clone(&calls),
                fail_after_delta: true,
            },
            policy,
        );
        let mut stream = provider.complete_stream(&(), &TestModel, &conversation, &[]);
        assert!(
            matches!(stream.next().await.unwrap().unwrap(), ProviderEvent::TextDelta(text) if text == "once")
        );
        assert!(stream.next().await.unwrap().is_err());
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

    struct CheckpointProvider(Arc<AtomicUsize>);
    impl Provider for CheckpointProvider {
        type Model = ();
        type Error = E;
        type Continuation = usize;
        fn supports_vision(&self, _: &()) -> bool {
            false
        }
        fn create_continuation(&self, _: &()) -> usize {
            7
        }
        async fn complete(
            &self,
            continuation: &usize,
            _: &(),
            _: &Conversation,
            _: &[ToolSpec],
        ) -> Result<(AssistantMessage, usize), E> {
            assert_eq!(*continuation, 7, "every retry sees the accepted checkpoint");
            let call = self.0.fetch_add(1, Ordering::SeqCst);
            if call < 2 {
                Err(E)
            } else {
                Ok((AssistantMessage::new("ok"), 8))
            }
        }
    }

    #[tokio::test]
    async fn retries_delegate_the_checkpoint_and_return_the_successor() {
        let provider = RetryProvider::with_policy(
            CheckpointProvider(Arc::new(AtomicUsize::new(0))),
            RetryPolicy {
                max_attempts: 3,
                initial_backoff: Duration::ZERO,
                max_backoff: Duration::ZERO,
                backoff_multiplier: 1.0,
            },
        );
        let continuation = provider.create_continuation(&());
        let (_, next) = provider
            .complete(&continuation, &(), &Conversation::new(), &[])
            .await
            .unwrap();
        assert_eq!(next, 8);
    }
}
