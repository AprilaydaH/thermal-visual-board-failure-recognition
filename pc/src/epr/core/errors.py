class EprError(Exception):
    """Base class for all application errors."""


class FrameSetError(EprError):
    """The frame set is unusable."""


class IncompleteFrameSetError(FrameSetError):
    """Channels are missing or do not match their descriptor."""


class ProtocolError(EprError):
    """The byte stream does not follow the acquisition protocol."""


class ChecksumError(ProtocolError):
    """Payload CRC does not match the header."""


class StorageError(EprError):
    """Raw data could not be stored."""
