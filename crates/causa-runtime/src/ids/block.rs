use causa_kernel::BlockId;

/// Allocate a fresh UUIDv7 fact-block identity.
///
/// The kernel accepts explicit IDs and does not provide randomness or time
/// access; runtime conveniences keep that capability at the outer layer.
pub fn new_block_id() -> BlockId {
    BlockId::new(uuid::Uuid::now_v7())
}
