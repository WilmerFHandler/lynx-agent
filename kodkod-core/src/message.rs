use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::sync::Arc;

use crate::{ToolCall, ToolResult};

/// A single image attachment in a user message.
///
/// Images are stored as raw bytes plus a MIME type; providers that support
/// vision are expected to encode them as base64 data URLs.
#[derive(Debug, Clone)]
pub struct Image {
    inner: Arc<ImageData>,
}

#[derive(Debug)]
struct ImageData {
    mime: String,
    data: Vec<u8>,
}

impl Image {
    pub fn new(mime: impl Into<String>, data: impl Into<Vec<u8>>) -> Self {
        Self {
            inner: Arc::new(ImageData {
                mime: mime.into(),
                data: data.into(),
            }),
        }
    }

    pub fn mime(&self) -> &str {
        &self.inner.mime
    }

    pub fn data(&self) -> &[u8] {
        &self.inner.data
    }

    /// Encode the image as a base64 data URL.
    pub fn to_data_url(&self) -> String {
        format!(
            "data:{};base64,{}",
            self.mime(),
            base64_bytes::encode(self.data())
        )
    }
}

impl PartialEq for Image {
    fn eq(&self, other: &Self) -> bool {
        self.mime() == other.mime() && self.data() == other.data()
    }
}

impl Eq for Image {}

impl Serialize for Image {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        ImageRef {
            mime: self.mime(),
            data: self.data(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Image {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let image = ImageWire::deserialize(deserializer)?;
        Ok(Self::new(image.mime, image.data))
    }
}

/// A document attachment in a user message.
///
/// Documents carry inline bytes only. Applications own persistence and any
/// provider-side upload lifecycle; providers decide which media types they can
/// send.
#[derive(Clone)]
pub struct Document {
    inner: Arc<DocumentData>,
}

#[derive(Debug)]
struct DocumentData {
    mime: String,
    filename: String,
    data: Vec<u8>,
}

impl Document {
    /// Construct a validated inline document.
    pub fn try_new(
        mime: impl Into<String>,
        filename: impl Into<String>,
        data: impl Into<Vec<u8>>,
    ) -> Result<Self, DocumentError> {
        let mime = mime.into();
        let filename = filename.into();
        let data = data.into();
        validate_mime(&mime)?;
        validate_filename(&filename)?;
        if data.is_empty() {
            return Err(DocumentError::EmptyData);
        }
        Ok(Self {
            inner: Arc::new(DocumentData {
                mime,
                filename,
                data,
            }),
        })
    }

    pub fn mime(&self) -> &str {
        &self.inner.mime
    }

    pub fn filename(&self) -> &str {
        &self.inner.filename
    }

    pub fn data(&self) -> &[u8] {
        &self.inner.data
    }

    /// Encode the document as a base64 data URL for providers with inline file input.
    pub fn to_data_url(&self) -> String {
        format!(
            "data:{};base64,{}",
            self.mime(),
            base64_bytes::encode(self.data())
        )
    }
}

impl fmt::Debug for Document {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Document")
            .field("mime", &self.mime())
            .field("filename", &self.filename())
            .field("byte_len", &self.data().len())
            .finish()
    }
}

impl PartialEq for Document {
    fn eq(&self, other: &Self) -> bool {
        self.mime() == other.mime()
            && self.filename() == other.filename()
            && self.data() == other.data()
    }
}

impl Eq for Document {}

impl Serialize for Document {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        DocumentRef {
            mime: self.mime(),
            filename: self.filename(),
            data: self.data(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Document {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let document = DocumentWire::deserialize(deserializer)?;
        Self::try_new(document.mime, document.filename, document.data)
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DocumentError {
    EmptyMime,
    InvalidMime,
    EmptyFilename,
    InvalidFilename,
    EmptyData,
}

impl fmt::Display for DocumentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyMime => f.write_str("document MIME type is empty"),
            Self::InvalidMime => f.write_str("document MIME type must be a valid type/subtype"),
            Self::EmptyFilename => f.write_str("document filename is empty"),
            Self::InvalidFilename => {
                f.write_str("document filename must not contain a path or control character")
            }
            Self::EmptyData => f.write_str("document data is empty"),
        }
    }
}

impl std::error::Error for DocumentError {}

#[derive(Serialize)]
struct DocumentRef<'a> {
    mime: &'a str,
    filename: &'a str,
    #[serde(with = "base64_bytes")]
    data: &'a [u8],
}

#[derive(Deserialize)]
struct DocumentWire {
    mime: String,
    filename: String,
    #[serde(with = "base64_bytes")]
    data: Vec<u8>,
}

fn validate_mime(mime: &str) -> Result<(), DocumentError> {
    if mime.trim().is_empty() {
        return Err(DocumentError::EmptyMime);
    }
    let Some((kind, subtype)) = mime.split_once('/') else {
        return Err(DocumentError::InvalidMime);
    };
    if kind.is_empty()
        || subtype.is_empty()
        || subtype.contains('/')
        || !kind.bytes().all(is_mime_token)
        || !subtype.bytes().all(is_mime_token)
    {
        return Err(DocumentError::InvalidMime);
    }
    Ok(())
}

fn is_mime_token(byte: u8) -> bool {
    byte.is_ascii_alphanumeric()
        || matches!(
            byte,
            b'!' | b'#' | b'$' | b'&' | b'^' | b'_' | b'.' | b'+' | b'-'
        )
}

fn validate_filename(filename: &str) -> Result<(), DocumentError> {
    if filename.trim().is_empty() {
        return Err(DocumentError::EmptyFilename);
    }
    if matches!(filename, "." | "..")
        || filename
            .chars()
            .any(|character| character.is_control() || matches!(character, '/' | '\\'))
    {
        return Err(DocumentError::InvalidFilename);
    }
    Ok(())
}

#[derive(Serialize)]
struct ImageRef<'a> {
    mime: &'a str,
    #[serde(with = "base64_bytes")]
    data: &'a [u8],
}

#[derive(Deserialize)]
struct ImageWire {
    mime: String,
    #[serde(with = "base64_bytes")]
    data: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UserMessage {
    content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    images: Vec<Image>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    documents: Vec<Document>,
    /// Whether this was injected mid-turn via steering rather than starting a new turn.
    #[serde(default, skip_serializing_if = "is_false")]
    steered: bool,
}

impl UserMessage {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            images: Vec::new(),
            documents: Vec::new(),
            steered: false,
        }
    }

    pub fn with_images(mut self, images: Vec<Image>) -> Self {
        self.images = images;
        self
    }

    pub fn with_documents(mut self, documents: Vec<Document>) -> Self {
        self.documents = documents;
        self
    }

    /// Mark this message as a steering injection (mid-turn, not a new turn).
    pub fn with_steered(mut self, steered: bool) -> Self {
        self.steered = steered;
        self
    }

    pub fn content(&self) -> &str {
        &self.content
    }

    pub fn images(&self) -> &[Image] {
        &self.images
    }

    pub fn documents(&self) -> &[Document] {
        &self.documents
    }

    pub fn steered(&self) -> bool {
        self.steered
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssistantMessage {
    content: String,
    tool_calls: Vec<ToolCall>,
}

impl AssistantMessage {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            tool_calls: Vec::new(),
        }
    }

    pub fn content(&self) -> &str {
        &self.content
    }

    pub fn tool_calls(&self) -> &[ToolCall] {
        &self.tool_calls
    }

    pub fn with_tool_calls(mut self, tool_calls: Vec<ToolCall>) -> Self {
        self.tool_calls = tool_calls;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemMessage {
    content: String,
}

impl SystemMessage {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
        }
    }

    pub fn content(&self) -> &str {
        &self.content
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "role")]
pub enum Message {
    #[serde(rename = "system")]
    System(SystemMessage),
    #[serde(rename = "user")]
    User(UserMessage),
    #[serde(rename = "assistant")]
    Assistant(AssistantMessage),
    #[serde(rename = "tool")]
    ToolResult(ToolResult),
}

impl Message {
    /// Tokenizer-free estimate of this message, including framing overhead.
    pub fn estimate_tokens(&self) -> u64 {
        crate::estimate::estimate_message(self)
    }
}

mod base64_bytes {
    use super::*;

    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    pub fn serialize<S: Serializer>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&encode(bytes))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
        let text = String::deserialize(deserializer)?;
        decode(&text).map_err(serde::de::Error::custom)
    }

    pub fn encode(input: &[u8]) -> String {
        let mut out = String::with_capacity((input.len() * 4).div_ceil(3));
        for chunk in input.chunks(3) {
            let mut buf = [0u8; 3];
            for (i, byte) in chunk.iter().enumerate() {
                buf[i] = *byte;
            }
            let triple = ((buf[0] as u32) << 16) | ((buf[1] as u32) << 8) | (buf[2] as u32);
            out.push(ALPHABET[((triple >> 18) & 0x3F) as usize] as char);
            out.push(ALPHABET[((triple >> 12) & 0x3F) as usize] as char);
            if chunk.len() > 1 {
                out.push(ALPHABET[((triple >> 6) & 0x3F) as usize] as char);
            } else {
                out.push('=');
            }
            if chunk.len() > 2 {
                out.push(ALPHABET[(triple & 0x3F) as usize] as char);
            } else {
                out.push('=');
            }
        }
        out
    }

    pub fn decode(input: &str) -> Result<Vec<u8>, String> {
        let bytes: Vec<u8> = input.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
        if bytes.is_empty() {
            return Ok(Vec::new());
        }
        if bytes.len() % 4 == 1 {
            return Err("invalid base64 length".to_string());
        }

        let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
        for chunk in bytes.chunks(4) {
            let mut buf = [0u8; 4];
            let mut padding = 0usize;
            for (i, &byte) in chunk.iter().enumerate() {
                if byte == b'=' {
                    padding += 1;
                    buf[i] = 0;
                } else {
                    if padding > 0 {
                        return Err("invalid base64 padding".to_string());
                    }
                    buf[i] = decode_char(byte)?;
                }
            }
            if padding > 2 {
                return Err("invalid base64 padding".to_string());
            }
            if chunk.len() < 4 && padding == 0 {
                return Err("invalid base64 length".to_string());
            }

            let triple = ((buf[0] as u32) << 18)
                | ((buf[1] as u32) << 12)
                | ((buf[2] as u32) << 6)
                | (buf[3] as u32);
            out.push(((triple >> 16) & 0xFF) as u8);
            if padding <= 1 {
                out.push(((triple >> 8) & 0xFF) as u8);
            }
            if padding == 0 {
                out.push((triple & 0xFF) as u8);
            }
        }
        Ok(out)
    }

    fn decode_char(byte: u8) -> Result<u8, String> {
        match byte {
            b'A'..=b'Z' => Ok(byte - b'A'),
            b'a'..=b'z' => Ok(byte - b'a' + 26),
            b'0'..=b'9' => Ok(byte - b'0' + 52),
            b'+' => Ok(62),
            b'/' => Ok(63),
            _ => Err(format!("invalid base64 character: {}", byte as char)),
        }
    }
}

fn is_false(value: &bool) -> bool {
    !value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_clones_share_the_immutable_payload() {
        let image = Image::new("image/png", vec![0x89, 0x50, 0x4e, 0x47]);
        let clone = image.clone();

        assert!(Arc::ptr_eq(&image.inner, &clone.inner));
        assert_eq!(image.to_data_url(), "data:image/png;base64,iVBORw==");
    }

    #[test]
    fn image_serialization_preserves_the_existing_wire_shape() {
        let image = Image::new("image/png", [0x89, 0x50]);

        let encoded = serde_json::to_value(&image).unwrap();
        assert_eq!(
            encoded,
            serde_json::json!({"mime": "image/png", "data": "iVA="})
        );
        assert_eq!(serde_json::from_value::<Image>(encoded).unwrap(), image);
    }

    #[test]
    fn document_clones_share_the_immutable_payload() {
        let document = Document::try_new("application/pdf", "notes.pdf", b"%PDF").unwrap();
        let clone = document.clone();

        assert!(Arc::ptr_eq(&document.inner, &clone.inner));
        assert_eq!(
            document.to_data_url(),
            "data:application/pdf;base64,JVBERg=="
        );
        assert_eq!(
            format!("{document:?}"),
            r#"Document { mime: "application/pdf", filename: "notes.pdf", byte_len: 4 }"#
        );
    }

    #[test]
    fn document_serialization_round_trips_validated_fields() {
        let document = Document::try_new("text/plain", "notes.txt", b"hello").unwrap();
        let encoded = serde_json::to_value(&document).unwrap();
        assert_eq!(
            encoded,
            serde_json::json!({"mime": "text/plain", "filename": "notes.txt", "data": "aGVsbG8="})
        );
        assert_eq!(
            serde_json::from_value::<Document>(encoded).unwrap(),
            document
        );
    }

    #[test]
    fn document_rejects_invalid_metadata_and_empty_bytes() {
        assert_eq!(
            Document::try_new("", "notes.txt", b"hello").unwrap_err(),
            DocumentError::EmptyMime
        );
        assert_eq!(
            Document::try_new("text plain", "notes.txt", b"hello").unwrap_err(),
            DocumentError::InvalidMime
        );
        assert_eq!(
            Document::try_new("text/plain", "../notes.txt", b"hello").unwrap_err(),
            DocumentError::InvalidFilename
        );
        assert_eq!(
            Document::try_new("text/plain", "notes.txt", Vec::new()).unwrap_err(),
            DocumentError::EmptyData
        );
    }

    #[test]
    fn document_deserialization_applies_constructor_validation() {
        let encoded =
            serde_json::json!({"mime": "text/plain", "filename": ".", "data": "aGVsbG8="});
        assert!(serde_json::from_value::<Document>(encoded).is_err());
    }
}
