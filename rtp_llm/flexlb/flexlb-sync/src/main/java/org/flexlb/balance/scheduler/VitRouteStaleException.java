package org.flexlb.balance.scheduler;

/** A selected ViT generation retired before the batch RPC handoff. */
final class VitRouteStaleException extends IllegalStateException {
    VitRouteStaleException(long requestId) {
        super("VIT_ROUTE_STALE: selected ViT worker retired for request " + requestId);
    }
}
