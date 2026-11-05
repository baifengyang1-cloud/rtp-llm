package org.flexlb.dao.master;

import org.flexlb.dao.route.RoleType;

import java.math.BigInteger;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/** Immutable, role-specific effective transfer contract. A null value denotes legacy. */
public record RoleCompatibility(
        int schemaVersion, String targetIdentity, Vision vision, Transfer transfer,
        Speculative speculative) {

    public record Vision(String identity, String embeddingContract) { }
    public record Speculative(String mode, String draftIdentity, int proposalWidth) { }
    public record Transfer(String backend, String keyFormat, long modelId, int tpSize,
                           int producerCpSize, int baseBlockTokens, List<Group> groups,
                           int workerTpSize, String producerLayout) {
        public Transfer { groups = groups.stream().sorted(java.util.Comparator.comparing(Group::tag)).toList(); }
    }
    public record Group(String tag, List<Integer> layers, String cacheKind, String dtype,
                        String framing, int blockTokens, int kernelBlockTokens,
                        int localKvHeads, List<Long> wireBytes, String projection, String representation) {
        public Group {
            layers = layers.stream().sorted().toList();
            wireBytes = List.copyOf(wireBytes);
        }
    }

    public void validate(RoleType role) {
        require(schemaVersion == 1, "unsupported compatibility schema");
        if (vision != null) {
            require(text(vision.identity()) && text(vision.embeddingContract()), "incomplete vision contract");
        }
        if (role == RoleType.VIT) {
            require(vision != null && !text(targetIdentity) && transfer == null && speculative == null,
                    "invalid ViT role contract");
            return;
        }
        require(role == RoleType.PREFILL || role == RoleType.DECODE || role == RoleType.PDFUSION,
                "unsupported compatibility role");
        require(text(targetIdentity) && speculative != null, "missing target/speculative contract");
        require(Set.of("none", "vanilla", "mtp", "eagle3", "eagle", "deterministic", "dspark")
                .contains(speculative.mode()), "unknown speculative mode");
        require(speculative.proposalWidth() > 0, "invalid proposal width");
        require(!speculative.mode().equals("none") || speculative.proposalWidth() == 1, "invalid non-speculative width");
        if (Set.of("vanilla", "mtp", "eagle3", "eagle", "dspark").contains(speculative.mode())) {
            require(text(speculative.draftIdentity()), "missing draft identity");
        }
        if (role == RoleType.PDFUSION && transfer == null) { return; }
        require(transfer != null, "missing KV transfer contract");
        require(Set.of("cache-store-rdma-v1", "cache-store-tcp-v1").contains(transfer.backend())
                        && "model-token-layer-tag-v1".equals(transfer.keyFormat()), "unknown transfer protocol");
        require(transfer.tpSize() > 0 && transfer.producerCpSize() > 0 && transfer.baseBlockTokens() > 0,
                "invalid transfer topology");
        require(transfer.workerTpSize() > 0 && Set.of("tp", "cp-replicated", "cp-sharded")
                .contains(transfer.producerLayout()), "invalid producer layout");
        require((transfer.producerCpSize() > 1) == transfer.producerLayout().equals("cp-sharded"),
                "inconsistent producer CP contract");
        require(!transfer.groups().isEmpty(), "missing transfer groups");
        Set<String> tags = new HashSet<>();
        for (Group g : transfer.groups()) {
            require(g.tag() != null && tags.add(g.tag()), "duplicate/missing group tag");
            require(!g.layers().isEmpty() && g.layers().stream().allMatch(l -> l >= 0)
                    && new HashSet<>(g.layers()).size() == g.layers().size(), "invalid group layers");
            require(Set.of("MultiHeadAttention", "MultiHeadLatentAttention", "LinearAttention", "OpaqueKV", "OpaqueState")
                    .contains(g.cacheKind()), "unknown cache kind");
            require(Set.of("fp16", "bf16", "fp32", "fp64", "int8", "uint8", "int16", "uint16",
                            "int32", "uint32", "int64", "uint64", "fp8_e4m3", "fp8_e8m0").contains(g.dtype())
                            && text(g.projection()) && text(g.representation())
                            && Set.of("kv", "split").contains(g.framing()),
                    "missing group layout/framing");
            require(g.blockTokens() > 0 && g.blockTokens() % transfer.baseBlockTokens() == 0
                    && g.kernelBlockTokens() > 0, "invalid block projection");
            require(g.localKvHeads() > 0 && g.wireBytes().size() == 4 && g.wireBytes().getFirst() > 0
                    && g.wireBytes().stream().allMatch(b -> b >= 0), "invalid wire layout");
        }
    }

    /** Pair matching is intentionally not transitive. Never close it transitively. */
    public static boolean matches(RoleType leftRole, RoleCompatibility left,
                                  RoleType rightRole, RoleCompatibility right) {
        if (left == null || right == null) { return left == right; }
        if (leftRole == RoleType.VIT || rightRole == RoleType.VIT) {
            return left.vision != null && left.vision.equals(right.vision);
        }
        if (!Objects.equals(left.targetIdentity, right.targetIdentity)
                || !Objects.equals(left.vision, right.vision)
                || !Objects.equals(left.speculative, right.speculative)) { return false; }
        Transfer p = leftRole == RoleType.DECODE ? right.transfer : left.transfer;
        Transfer d = leftRole == RoleType.DECODE ? left.transfer : right.transfer;
        if (p == null || d == null || !p.backend().equals(d.backend())
                || !p.keyFormat().equals(d.keyFormat()) || p.modelId() != d.modelId()
                || !p.producerLayout().equals(d.producerLayout())
                || p.producerCpSize() != d.producerCpSize() || p.baseBlockTokens() != d.baseBlockTokens()
                || Math.max(p.workerTpSize(), d.workerTpSize()) % Math.min(p.workerTpSize(), d.workerTpSize()) != 0
                || Math.max(p.tpSize(), d.tpSize()) % Math.min(p.tpSize(), d.tpSize()) != 0
                || p.groups().size() != d.groups().size()) { return false; }
        for (int i = 0; i < p.groups().size(); i++) {
            Group a = p.groups().get(i);
            Group b = d.groups().get(i);
            if (!a.tag().equals(b.tag()) || !a.layers().equals(b.layers())
                    || !a.cacheKind().equals(b.cacheKind()) || !a.dtype().equals(b.dtype())
                    || !a.framing().equals(b.framing()) || !a.projection().equals(b.projection())
                    || !a.representation().equals(b.representation())
                    || a.blockTokens() != b.blockTokens() || a.kernelBlockTokens() != b.kernelBlockTokens()) {
                return false;
            }
            boolean partitioned = a.cacheKind().equals("MultiHeadAttention");
            if (partitioned && (long) a.localKvHeads() * p.tpSize() != (long) b.localKvHeads() * d.tpSize()) {
                return false;
            }
            // cache-store opaque/hybrid blocks have no TP slicing path.
            if (partitioned && a.framing().equals("kv") && p.tpSize() != d.tpSize()) { return false; }
            boolean headSlicing = partitioned && !p.producerLayout().equals("cp-sharded");
            if (partitioned && !headSlicing
                    && (p.tpSize() != d.tpSize() || a.localKvHeads() != b.localKvHeads())) { return false; }
            long aDivisor = headSlicing ? a.localKvHeads() : 1;
            long bDivisor = headSlicing ? b.localKvHeads() : 1;
            for (int j = 0; j < 4; j++) {
                if (!BigInteger.valueOf(a.wireBytes().get(j)).multiply(BigInteger.valueOf(bDivisor))
                        .equals(BigInteger.valueOf(b.wireBytes().get(j)).multiply(BigInteger.valueOf(aDivisor)))) {
                    return false;
                }
            }
        }
        return true;
    }

    private static boolean text(String value) { return value != null && !value.isBlank(); }
    private static void require(boolean condition, String message) {
        if (!condition) { throw new IllegalArgumentException(message); }
    }
}
