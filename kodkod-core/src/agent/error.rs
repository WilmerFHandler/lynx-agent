use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub enum AgentError<E> {
    Provider(E),
    /// A provider stream ended without its authoritative completion event.
    ProviderStreamEnded,
    MaxToolRoundsExceeded {
        max: usize,
    },
    /// The caller requested cancellation via [`TaskControl`](super::TaskControl).
    Cancelled,
    /// The one-execution control was cancelled or previously used.
    ControlAlreadyUsed,
}

impl<E> fmt::Display for AgentError<E>
where
    E: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Provider(error) => write!(f, "provider failed: {error}"),
            Self::ProviderStreamEnded => {
                write!(f, "provider stream ended before completing the response")
            }
            Self::MaxToolRoundsExceeded { max } => {
                write!(f, "assistant requested tools for more than {max} rounds")
            }
            Self::Cancelled => write!(f, "agent run was cancelled"),
            Self::ControlAlreadyUsed => write!(f, "task control was cancelled or already used"),
        }
    }
}

impl<E> Error for AgentError<E>
where
    E: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Provider(error) => Some(error),
            Self::ProviderStreamEnded
            | Self::MaxToolRoundsExceeded { .. }
            | Self::Cancelled
            | Self::ControlAlreadyUsed => None,
        }
    }
}
