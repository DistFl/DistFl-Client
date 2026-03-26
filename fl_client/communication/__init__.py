"""Communication module — serialization and compression."""
from fl_client.communication.compressor import compress, decompress
from fl_client.communication.serializer import deserialize_weights, serialize_weights

__all__ = ["compress", "decompress", "serialize_weights", "deserialize_weights"]
