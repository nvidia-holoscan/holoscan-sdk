# Holoscan C++ API

- [Holoscan C++ API](#holoscan-c-api)
  - [Namespaces](#namespaces)
  - [Macros](#macros)
    - [Operator Definition](#operator-definition)
    - [Resource Definition](#resource-definition)
    - [Condition Definition](#condition-definition)
    - [Scheduler Definition](#scheduler-definition)
    - [Logging](#logging)
  - [Classes](#classes)
    - [Core](#core)
    - [Operators](#operators)
    - [GXF Components](#gxf-components)
      - [Conditions](#conditions)
      - [Resources](#resources)
      - [Schedulers](#schedulers)
      - [Network Contexts](#network-contexts)
    - [Native Operator Support](#native-operator-support)
    - [Analytics](#analytics)
    - [Data Logging](#data-logging)
    - [Domain Objects](#domain-objects)
      - [Tensor (interoperability with GXF Tensor and DLPack interface)](#tensor-interoperability-with-gxf-tensor-and-dlpack-interface)
        - [Class/Struct](#classstruct)
        - [Functions](#functions)
    - [Utilities](#utilities)
      - [Measurement](#measurement)
  - [Enums](#enums)
  - [Functions](#functions-1)
  - [Typedefs](#typedefs)
  - [Variables](#variables)

## Namespaces

- {ref}`namespace_holoscan__gxf`
- {ref}`namespace_holoscan__ops`

## Macros

### Operator Definition

- {ref}`exhale_define_operator_8hpp_1ab2c635a927962650e72a33623f2f9ca1`
- {ref}`exhale_define_operator_8hpp_1af59d84ffa537c4b1186e2a1ae2be30ad`
- {ref}`exhale_define_gxf__codelet_8hpp_1adb58640018e9787efd52475fc95a958e`

### Resource Definition

- {ref}`exhale_define_resource_8hpp_1a4c671dac9ff91b8ef6f9b5a5a168941f`
- {ref}`exhale_define_resource_8hpp_1a94bcc7c12f51de26c6873cf1e7be9ea9`
- {ref}`exhale_define_gxf__component__resource_8hpp_1a269b593e54aca3766ff5b26f780f3e35`

### Condition Definition

- {ref}`exhale_define_condition_8hpp_1a1cc440f7187549071aba2c2703265a2d`
- {ref}`exhale_define_condition_8hpp_1a4c8ccc5264282a21b4ae2308dc1515d6`

### Scheduler Definition

- {ref}`exhale_define_scheduler_8hpp_1a734c868ea263966370d6773ba4f617fa`
- {ref}`exhale_define_scheduler_8hpp_1a003acb63ae306b2971f5508061657317`

### Logging

- {ref}`exhale_define_logger_8hpp_1a7e3138f9692735dc846a302a58057c6c`
- {ref}`exhale_define_logger_8hpp_1a3cc81037bfb59885b17af859a383d1dd`
- {ref}`exhale_define_logger_8hpp_1aae5d745102f8830c1dd8cc10bca8c4bd`
- {ref}`exhale_define_logger_8hpp_1a9c1127341727e5b368b8e65248f41b9c`
- {ref}`exhale_define_logger_8hpp_1a479748f09598bf5412b305bcfdd91340`
- {ref}`exhale_define_logger_8hpp_1ab9563f3d4ce1dfc852c3a034060aa8a7`

## Classes

### Core

- {ref}`exhale_class_classholoscan_1_1Application`
- {ref}`exhale_class_classholoscan_1_1Arg`
- {ref}`exhale_class_classholoscan_1_1ArgList`
- {ref}`exhale_class_classholoscan_1_1ArgType`
- {ref}`exhale_class_classholoscan_1_1ArgumentSetter`
- {ref}`exhale_struct_structholoscan_1_1CLIOptions`
- {ref}`exhale_class_classholoscan_1_1Component`
- {ref}`exhale_class_classholoscan_1_1ComponentSpec`
- {ref}`exhale_class_classholoscan_1_1Condition`
- {ref}`exhale_class_classholoscan_1_1Config`
- {ref}`exhale_class_classholoscan_1_1DataFlowTracker`
- {ref}`exhale_class_classholoscan_1_1ExecutionContext`
- {ref}`exhale_class_classholoscan_1_1ExtensionManager`
- {ref}`exhale_class_classholoscan_1_1Executor`
- {ref}`exhale_class_classholoscan_1_1FlowGraph`
- {ref}`exhale_class_classholoscan_1_1Fragment`
- {ref}`exhale_class_classholoscan_1_1Graph`
- {ref}`exhale_class_classholoscan_1_1InputContext`
- {ref}`exhale_class_classholoscan_1_1IOSpec`
- {ref}`exhale_class_classholoscan_1_1MessageLabel`
- {ref}`exhale_class_classholoscan_1_1MetadataDictionary`
- {ref}`exhale_class_classholoscan_1_1MetaParameter`
- {ref}`exhale_class_classholoscan_1_1Operator`
- {ref}`exhale_class_classholoscan_1_1OperatorSpec`
- {ref}`exhale_struct_structholoscan_1_1OperatorTimestampLabel`
- {ref}`exhale_class_classholoscan_1_1OutputContext`
- {ref}`exhale_class_classholoscan_1_1ParameterWrapper`
- {ref}`exhale_class_classholoscan_1_1Resource`
- {ref}`exhale_class_classholoscan_1_1Scheduler`
- {ref}`exhale_class_classholoscan_1_1ThreadPool`

### Operators

- {ref}`exhale_class_classholoscan_1_1ops_1_1AsyncPingRxOp`
- {ref}`exhale_class_classholoscan_1_1ops_1_1AsyncPingTxOp`
- {ref}`exhale_class_classholoscan_1_1ops_1_1BayerDemosaicOp`
- {ref}`exhale_class_classholoscan_1_1ops_1_1FormatConverterOp`
- {ref}`exhale_class_classholoscan_1_1ops_1_1GXFCodeletOp`
- {ref}`exhale_class_classholoscan_1_1ops_1_1HolovizOp`
- {ref}`exhale_class_classholoscan_1_1ops_1_1InferenceOp`
- {ref}`exhale_class_classholoscan_1_1ops_1_1InferenceProcessorOp`
- {ref}`exhale_class_classholoscan_1_1ops_1_1PingRxOp`
- {ref}`exhale_class_classholoscan_1_1ops_1_1PingTensorRxOp`
- {ref}`exhale_class_classholoscan_1_1ops_1_1PingTensorTxOp`
- {ref}`exhale_class_classholoscan_1_1ops_1_1PingTxOp`
- {ref}`exhale_class_classholoscan_1_1ops_1_1SegmentationPostprocessorOp`
- {ref}`exhale_class_classholoscan_1_1ops_1_1V4L2VideoCaptureOp`
- {ref}`exhale_class_classholoscan_1_1ops_1_1VideoStreamRecorderOp`
- {ref}`exhale_class_classholoscan_1_1ops_1_1VideoStreamReplayerOp`
- {ref}`exhale_struct_structholoscan_1_1ops_1_1BufferInfo`
- {ref}`exhale_struct_structholoscan_1_1ops_1_1HolovizOp_1_1InputSpec`
- {ref}`exhale_struct_structholoscan_1_1ops_1_1HolovizOp_1_1InputSpec_1_1View`
- {ref}`exhale_struct_structholoscan_1_1codec_3_01ops_1_1HolovizOp_1_1InputSpec_01_4`
- {ref}`exhale_struct_structholoscan_1_1codec_3_01ops_1_1HolovizOp_1_1InputSpec_1_1View_01_4`
- {ref}`exhale_struct_structholoscan_1_1codec_3_01std_1_1vector_3_01ops_1_1HolovizOp_1_1InputSpec_01_4_01_4`
- {ref}`exhale_struct_structholoscan_1_1codec_3_01std_1_1vector_3_01ops_1_1HolovizOp_1_1InputSpec_1_1View_01_4_01_4`
- {ref}`exhale_struct_structholoscan_1_1ops_1_1InferenceOp_1_1DataMap`
- {ref}`exhale_struct_structholoscan_1_1ops_1_1InferenceOp_1_1DataVecMap`
- {ref}`exhale_struct_structholoscan_1_1ops_1_1InferenceProcessorOp_1_1DataMap`
- {ref}`exhale_struct_structholoscan_1_1ops_1_1InferenceProcessorOp_1_1DataVecMap`
- {ref}`exhale_struct_structholoscan_1_1ops_1_1V4L2VideoCaptureOp_1_1Buffer`

### GXF Components

#### Conditions

- {ref}`exhale_class_classholoscan_1_1AsynchronousCondition`
- {ref}`exhale_class_classholoscan_1_1BooleanCondition`
- {ref}`exhale_class_classholoscan_1_1CountCondition`
- {ref}`exhale_class_classholoscan_1_1CudaBufferAvailableCondition`
- {ref}`exhale_class_classholoscan_1_1CudaEventCondition`
- {ref}`exhale_class_classholoscan_1_1CudaStreamCondition`
- {ref}`exhale_class_classholoscan_1_1DownstreamMessageAffordableCondition`
- {ref}`exhale_class_classholoscan_1_1ExpiringMessageAvailableCondition`
- {ref}`exhale_class_classholoscan_1_1MessageAvailableCondition`
- {ref}`exhale_class_classholoscan_1_1PeriodicCondition`

#### Resources

- {ref}`exhale_class_classholoscan_1_1Allocator`
- {ref}`exhale_class_classholoscan_1_1BlockMemoryPool`
- {ref}`exhale_class_classholoscan_1_1Clock`
- {ref}`exhale_class_classholoscan_1_1CudaAllocator`
- {ref}`exhale_class_classholoscan_1_1CudaStreamPool`
- {ref}`exhale_class_classholoscan_1_1DoubleBufferReceiver`
- {ref}`exhale_class_classholoscan_1_1DoubleBufferTransmitter`
- {ref}`exhale_class_classholoscan_1_1GXFComponentResource`
- {ref}`exhale_class_classholoscan_1_1ManualClock`
- {ref}`exhale_class_classholoscan_1_1RealtimeClock`
- {ref}`exhale_class_classholoscan_1_1Receiver`
- {ref}`exhale_class_classholoscan_1_1RMMAllocator`
- {ref}`exhale_class_classholoscan_1_1SerializationBuffer`
- {ref}`exhale_class_classholoscan_1_1StdComponentSerializer`
- {ref}`exhale_class_classholoscan_1_1StdEntitySerializer`
- {ref}`exhale_class_classholoscan_1_1StreamOrderedAllocator`
- {ref}`exhale_class_classholoscan_1_1ThreadPool`
- {ref}`exhale_class_classholoscan_1_1Transmitter`
- {ref}`exhale_class_classholoscan_1_1UcxComponentSerializer`
- {ref}`exhale_class_classholoscan_1_1UcxEntitySerializer`
- {ref}`exhale_class_classholoscan_1_1UcxReceiver`
- {ref}`exhale_class_classholoscan_1_1UcxSerializationBuffer`
- {ref}`exhale_class_classholoscan_1_1UcxTransmitter`
- {ref}`exhale_class_classholoscan_1_1UnboundedAllocator`

#### Schedulers

- {ref}`exhale_class_classholoscan_1_1EventBasedScheduler`
- {ref}`exhale_class_classholoscan_1_1GreedyScheduler`
- {ref}`exhale_class_classholoscan_1_1MultiThreadScheduler`

#### Network Contexts

- {ref}`exhale_class_classholoscan_1_1UcxContext`

### Native Operator Support

- {ref}`exhale_class_classholoscan_1_1Message`

### Analytics

- {ref}`exhale_class_classholoscan_1_1CsvDataExporter`
- {ref}`exhale_class_classholoscan_1_1DataExporter`

### Data Logging

- {ref}`exhale_class_classholoscan_1_1DataLogger`
- {ref}`exhale_class_classholoscan_1_1DataLoggerResource`
- {ref}`exhale_class_classholoscan_1_1data__loggers_1_1BasicConsoleLogger`
- {ref}`exhale_class_classholoscan_1_1data__loggers_1_1SimpleTextSerializer`

### Domain Objects

#### Tensor (interoperability with GXF Tensor and DLPack interface)

##### Class/Struct

- {ref}`exhale_class_classholoscan_1_1Tensor`
- {ref}`exhale_class_classholoscan_1_1TensorMap`

##### Type Definitions

- {ref}`exhale_typedef_tensor_8hpp_1afd6b2b681b22ddeaca73cd6a6232c5e1`
- {ref}`exhale_typedef_tensor_8hpp_1ad758981759a8dd0f69b807ad98451af4`

##### Functions

- {ref}`exhale_function_tensor_8hpp_1aba4ddc93980bd147bc4970bfa3ff4d81`
- {ref}`exhale_function_tensor_8hpp_1a556c5cb30a8df020398c43aadb4f0922`
- {ref}`exhale_function_tensor_8hpp_1a44e273bf355e7a145e76756817b92f68`
- {ref}`exhale_function_tensor_8hpp_1aa973c4fccb61338f25cfb0ee4a272b83`

### Utilities

#### Measurement

- {ref}`exhale_class_classholoscan_1_1Timer`

## Enums

- {ref}`exhale_enum_allocator_8hpp_1a8b7f69b7437dab3499a14e35a5d72c75`
- {ref}`exhale_enum_arg_8hpp_1a797fe63fdfc22e0260d6d44b85f0d9f9`
- {ref}`exhale_enum_arg_8hpp_1af4a95575587845846f8c58fa49bab5ab`
- {ref}`exhale_enum_condition_8hpp_1a5dc906177a4609bd59caa475ba7cdb30`
- {ref}`exhale_enum_dataflow__tracker_8hpp_1ad33aa68261c043b54c11a39337f5ec1f`
- {ref}`exhale_enum_errors_8hpp_1a33ad7eb800d03ecad89d263d61891a21`
- {ref}`exhale_enum_logger_8hpp_1aa521e133a102a486c72b613570749983`
- {ref}`exhale_enum_metadata_8hpp_1acc04c5458cb739cced68ea01df97f7e4`
- {ref}`exhale_enum_parameter_8hpp_1aa1004e0a8386455dbece91f902e656a9`
- {ref}`exhale_enum_scheduler_8hpp_1ae182dbb0da5dc05fc4661bec4ee09dc5`

### Operator-Specific Enums

- {ref}`exhale_enum_format__converter_8hpp_1ac05fa06405b47ecc7d71b02d5256966c`
- {ref}`exhale_enum_format__converter_8hpp_1acee968a280efc1cd6985d571358cf36b`

### Inference Module Enums

- {ref}`exhale_enum_holoinfer__constants_8hpp_1a22febad39220b4b17ce7afa6fa59d15c`
- {ref}`exhale_enum_holoinfer__constants_8hpp_1a253286b72a22ac4e1128061e3ceba4f6`
- {ref}`exhale_enum_holoinfer__constants_8hpp_1a4cf747d00d17adc861a963ca55e9ded4`
- {ref}`exhale_enum_holoinfer__constants_8hpp_1a654bb7758997dfd1b8a03014c5dfae4a`

### Visualization Module Enums

- {ref}`exhale_enum_depth__map__render__mode_8hpp_1a1610f620d0ec7bb602c2f05da951e4fa`
- {ref}`exhale_enum_image__format_8hpp_1ac1df8331638c94daefdc3b27385e0433`
- {ref}`exhale_enum_image__format_8hpp_1ada3d391622462f8571238aec0ea420ec`
- {ref}`exhale_enum_init__flags_8hpp_1aabff63dc2d78808d2a1e55f651be95bb`
- {ref}`exhale_enum_primitive__topology_8hpp_1a0bd75c25db987e4b90d482f641883a35`

## Functions

- {ref}`exhale_function_application_8hpp_1afd5a542cf1c8166e8c32bd2618adfd71`
- {ref}`exhale_function_logger_8hpp_1ab96dd0720200f30d82a68125a9880930`
- {ref}`exhale_function_logger_8hpp_1a4ff4b7f522e771a54df59d2cea8dc4e6`
- {ref}`exhale_function_logger_8hpp_1a2a6bf3f4a33139e6340f3242e198d994`
- {ref}`exhale_function_logger_8hpp_1a511fe905cb7b47a6abc589544f0dd007`
- {ref}`exhale_function_logger_8hpp_1af5d15f6a4c76ce8883fb9a3df64cf8af`
- {ref}`exhale_function_logger_8hpp_1a3c9f3f0113a317dadcbfd9c402f04882`
- {ref}`exhale_function_logger_8hpp_1a60d44225f9b1825fd9884879fee86db3`
- {ref}`exhale_function_logger_8hpp_1a87d3a6c0dc0d1186f82ed4b9666bd852`
- {ref}`exhale_function_logger_8hpp_1a30f98b4f124293d8fe8493e6de2c06d0`
- {ref}`exhale_function_logger_8hpp_1a8017df47ff9679f3805d1a0b7e6dfe94`
- {ref}`exhale_function_gxf__io__context_8hpp_1a629d0779f82fee87ce6152b7ab62ee02`
- {ref}`exhale_function_io__context_8hpp_1a24d0b14ea7aaf76c7505e5f3b707afc8`

## Typedefs

- {ref}`exhale_typedef_expected_8hpp_1a21751b7658eb9233f58d9a5a4f2c1bb3`
- {ref}`exhale_typedef_expected_8hpp_1a8c29243ec7fdd4aef7d935f0c72dc3f2`
- {ref}`exhale_typedef_expected_8hpp_1add9f49110f4c6595e76137d1481cc95e`
- {ref}`exhale_typedef_expected_8hpp_1af408adac7b395bb6c10178620a7c8bf9`
- {ref}`exhale_typedef_flow__graph_8hpp_1a4aecea229722688be3a7d30348421aa3`
- {ref}`exhale_typedef_flow__graph_8hpp_1aeb18a0625c0375e8da1814e89670c608`
- {ref}`exhale_typedef_forward__def_8hpp_1acaccb6c50efc493a58bf447d50bf0164`
- {ref}`exhale_typedef_graph_8hpp_1a302b71634787b2078f85ce402ff3f47e`
- {ref}`exhale_typedef_graph_8hpp_1a9fa5bdfec4d57e402a8deffe61750b36`
- {ref}`exhale_typedef_graph_8hpp_1ab20b0666014a0fc634583c1dc5af455e`
- {ref}`exhale_typedef_graph_8hpp_1ae1a8c1bff7a3db3c062684612e1e050c`
- {ref}`exhale_typedef_graph_8hpp_1ae5b60650556412963c694a9b15a81cc1`
- {ref}`exhale_typedef_graph_8hpp_1afb2c17034d7e3e004a9131763e8f3d5f`
- {ref}`exhale_typedef_metadata_8hpp_1a4fc913c666f8b7a1afc6b0d0f1e0aa8c`
- {ref}`exhale_typedef_type__traits_8hpp_1a0d3f2b03e9f4324fd70e2cd5139fd1ec`
- {ref}`exhale_typedef_type__traits_8hpp_1a101b19fbb6e8792e899bbd1fff211698`
- {ref}`exhale_typedef_type__traits_8hpp_1a2e7271fe8cdf8230122ceb983228ec4e`
- {ref}`exhale_typedef_type__traits_8hpp_1a399cd74a5f909d7b3815158ac32c8471`
- {ref}`exhale_typedef_type__traits_8hpp_1a737b5fa7c80def8d7ee21a775b7ef00a`
- {ref}`exhale_typedef_type__traits_8hpp_1a8898228ed7bb54554abcb87c4d2b8f7c`
- {ref}`exhale_typedef_type__traits_8hpp_1ab095d25df2246a0edf7d95b99b728908`
- {ref}`exhale_typedef_type__traits_8hpp_1aef13568514a360858861221b587da84e`

### Inference Module Typedefs

- {ref}`exhale_typedef_data__processor_8hpp_1aebc6df65b363c69857e1a735ea8108ce`
- {ref}`exhale_typedef_holoinfer__buffer_8hpp_1a087e5c16b34b9ed56caef479b684c421`
- {ref}`exhale_typedef_holoinfer__buffer_8hpp_1a33b28575b822fc2e74dd30eab1ae22bf`
- {ref}`exhale_typedef_holoinfer__buffer_8hpp_1acafba819f141eab27791da813db442db`
- {ref}`exhale_typedef_holoinfer__buffer_8hpp_1aeff0d061b611008ee9ba3dea8e1f167d`
- {ref}`exhale_typedef_holoinfer__constants_8hpp_1a570844f644ca51f733f6420860f16338`
- {ref}`exhale_typedef_holoinfer__constants_8hpp_1ab4984b33b9a16734f72d72ce45e4a4ba`
- {ref}`exhale_typedef_modules_2holoinfer_2src_2include_2holoinfer__utils_8hpp_1a45cf3d8d57bd6bbf4de6fcfcb24f2de9`

### Visualization Module Typedefs

- {ref}`exhale_typedef_modules_2holoviz_2src_2holoviz_2holoviz_8hpp_1a3c96d80d363e67d13a41b5d1821f3242`
- {ref}`exhale_typedef_modules_2holoviz_2src_2holoviz_2holoviz_8hpp_1a4dc626cd976f39a19971569b6727a3a0`

## Variables

- {ref}`exhale_variable_cpu__resource__monitor_8hpp_1a81f044dd96503e8919462ec079ecbc65`
- {ref}`exhale_variable_dataflow__tracker_8hpp_1a3e240487edfc73e59755f29d5a9ffe78`
- {ref}`exhale_variable_dataflow__tracker_8hpp_1a7cb835d2f1d0cad83d925f0b44d13acd`
- {ref}`exhale_variable_dataflow__tracker_8hpp_1a7e23da7f0ffa20bc79bdb743564bf0f6`
- {ref}`exhale_variable_dataflow__tracker_8hpp_1a920aecafd8024abe3a125cbf0e32ce63`
- {ref}`exhale_variable_dataflow__tracker_8hpp_1aa2cfa7e17c78f18fc91c444dfa0945d1`
- {ref}`exhale_variable_dataflow__tracker_8hpp_1af7cd18b9eb2b9d9b76d328b59900f566`
- {ref}`exhale_variable_expected_8hpp_1ae6efc4444700a9a08911d884857bb06c`
- {ref}`exhale_variable_gpu__resource__monitor_8hpp_1ae9c4ec64e9b50146f256c3e70eccb823`
- {ref}`exhale_variable_io__context_8hpp_1a7d68812a7241b94af93ec46784458585`
- {ref}`exhale_variable_serialization__buffer_8hpp_1aa7a8ceba3b1b28fd04e0139b78701b36`
- {ref}`exhale_variable_type__traits_8hpp_1a2b61ac0c36bd39ca398dde9664e65e33`
- {ref}`exhale_variable_type__traits_8hpp_1a3891b0c8d38e9c0a11b23dc8edd31ceb`
- {ref}`exhale_variable_type__traits_8hpp_1a7c08bbb1ef7ef321fb992a95efd512da`
- {ref}`exhale_variable_type__traits_8hpp_1ab85cb33786104651460327900b9a4bb0`
- {ref}`exhale_variable_type__traits_8hpp_1ad783526c0f45be5263f0e3a05593d611`
- {ref}`exhale_variable_type__traits_8hpp_1ad7d7f0199299096140f9cfb74300d0de`
- {ref}`exhale_variable_type__traits_8hpp_1af646bfc4bc70953fc1efc7bc3db459a5`
- {ref}`exhale_variable_type__traits_8hpp_1af73fdd04b98b6ee3860a13bfe81229fb`
- {ref}`exhale_variable_ucx__receiver_8hpp_1aef73979384b300f441fb9c0c9dec557e`
- {ref}`exhale_variable_ucx__serialization__buffer_8hpp_1a562f8204ee23b5237632895651668eb8`

### Inference Module Variables

- {ref}`exhale_variable_infer__manager_8hpp_1a25cde569b0d251fbd30765ec68766a0b`
- {ref}`exhale_variable_infer__manager_8hpp_1a52921e7945bc7ee74cb281271e8fbeb4`
- {ref}`exhale_variable_modules_2holoinfer_2src_2include_2holoinfer__utils_8hpp_1aed7f62ec8a46ab6cbe3334ac26c719c6`
- {ref}`exhale_variable_utils_8hpp_1aba4496e4cd0c7966ca1730727c109373`

```{toctree}
:maxdepth: 2

cpp/apidoc_root
```
