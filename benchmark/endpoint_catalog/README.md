# Endpoint Catalog Benchmark

This source-tree Google Benchmark suite measures cold endpoint-catalog
construction and normalization. It covers:

- public `ChannelEndpointCatalog::Builder` construction plus final canonical
  build for 1, 8, 64, 1024, and the schema-v1 maximum of 4096 endpoints; and
- complete POD layout validation, digest derivation, and canonical catalog
  materialization for 1, 8, 64, and the schema-v1 maximum of 1024 fields.

The endpoint fixtures and POD draft copies are prepared outside the timed
regions. Reported Builder rows include the endpoint record copies because those
are part of the public authoring operation. CTest runs a short smoke test and
does not impose timing thresholds.

Run the suite from the repository root:

```sh
./public/run bench --name endpoint_catalog
```

For repeated characterization and JSON output:

```sh
./public/run bench --stable-system --name endpoint_catalog \
  --output-json endpoint-catalog-benchmark.json \
  --benchmark-args \
  "--benchmark_min_time=0.01s --benchmark_repetitions=5 \
   --benchmark_report_aggregates_only=true"
```

These are compile/startup-path measurements, not Channel data-path latency.
The 1024/4096 rows are scaling sentinels for admitted format limits; small rows
better represent common application catalogs.
