# Third-party descriptor-backed Tensor schema

This example defines `app3p.DescriptorFrame`, an application-owned FlatBuffers root with ordinary
metadata plus one `holoscan.schema.Tensor`. It is deliberately outside the SDK namespace and builds
against either the SDK build tree or an installed SDK package.

The CMake file uses the installed `holoscan_add_flatbuffer_schema()` helper and
the SDK's `tensor.fbs`. The reflected direct Tensor field selects descriptor mode and generates
`descriptor_frame_schema_traits.hpp`: identity, bounded codec policy, the root and native traits, indexed
Tensor-view accessors, the ordinary-field codec, and the
`make_standard_descriptor_tensor_port_type_ops()` factory call. The SDK default adapter owns
persistent sample storage, bounded metadata-only Tensor encoding, Tensor/root consistency checks,
Tensor assignment, and sample metadata bookkeeping. The generated codec maps scalar, string,
scalar-vector, string-vector, and nested-table fields in this example. The executable then proves
the intended application path for `Output<DescriptorFrameT>::allocate_tensor()` and
`emit_tensor()` through a compiled graph.

## Application path

The user-facing application path is `Source` and `Sink`: declare
`Output<app3p::DescriptorFrameT>` / `Input<app3p::DescriptorFrameT>`, configure Tensor contracts,
allocate and populate an output-owned Tensor, call `emit_tensor()`, connect the ports with
`Graph::add_flow()`, compile, and call `run_async()`.

The small `Completion` helper is the finite application's normal shutdown condition. The public
CTest is intentionally a compile-and-run smoke test.

Applications do not inspect `RuntimePortTypeOps`, manually prepare sample storage, or assert
internal plan-record counts in order to use a generated descriptor schema.

## Reflection and field ordering

Descriptor mode asks the installed pinned `flatc` for a binary schema and decodes it with the
matching installed `flatbuffers/reflection.fbs`. Tensor fields are selected only when their exact
reflected type is `holoscan.schema.Tensor`, then ordered by FlatBuffers field `id`. Descriptor mode
requires explicit IDs throughout the schema closure so declaration order cannot become a second
ordering convention.

The generated root binding retains both `tensor_fields` and `tensor_field_ids`. Runtime admission
uses the complete reflected Tensor-field inventory,
ordered by FlatBuffers field ID; multi-field schemas receive one bounded provider per authoritative
field. See the
[multi-field bundle and lifecycle-rate guidance](../../../docs/custom_flatbuffer_schemas.mdx#bound-lifecycle-selected-multi-field-rates).

## Custom-codec escape hatch

Generation fails with a schema path and field name for unsupported unions, table/struct vectors,
deprecated fields, or other unsupported object-API shapes. For one of those roots,
`DESCRIPTOR_TENSOR_CUSTOM_CODEC` retains generated identity and Tensor inventory while a
hand-written codec calls `make_standard_descriptor_tensor_port_type_ops()`. The standard adapter
remains the only storage and transport implementation; the escape hatch changes only ordinary-field
mapping and adds no registry. Descriptor generation also emits the narrow
cross-namespace `CreateTensor` forwarding overload needed by FlatBuffers 25.12.19.
