��
��
�
ArgMax

input"T
	dimension"Tidx
output"output_type"!
Ttype:
2	
"
Tidxtype0:
2	"
output_typetype0	:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
�
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
�
RandomUniformInt

shape"T
minval"Tout
maxval"Tout
output"Tout"
seedint "
seed2int "
Touttype:
2	"
Ttype:
2	�
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.6.32v2.6.2-194-g92a6bb065498��
d
VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0	
�
%QNetwork/EncodingNetwork/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*6
shared_name'%QNetwork/EncodingNetwork/dense/kernel
�
9QNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOpReadVariableOp%QNetwork/EncodingNetwork/dense/kernel*
_output_shapes

:d*
dtype0
�
#QNetwork/EncodingNetwork/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*4
shared_name%#QNetwork/EncodingNetwork/dense/bias
�
7QNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOpReadVariableOp#QNetwork/EncodingNetwork/dense/bias*
_output_shapes
:d*
dtype0
�
QNetwork/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*(
shared_nameQNetwork/dense_1/kernel
�
+QNetwork/dense_1/kernel/Read/ReadVariableOpReadVariableOpQNetwork/dense_1/kernel*
_output_shapes

:d*
dtype0
�
QNetwork/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameQNetwork/dense_1/bias
{
)QNetwork/dense_1/bias/Read/ReadVariableOpReadVariableOpQNetwork/dense_1/bias*
_output_shapes
:*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
T

train_step
metadata
model_variables
_all_assets

signatures
CA
VARIABLE_VALUEVariable%train_step/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
	3


0
1
 
ge
VARIABLE_VALUE%QNetwork/EncodingNetwork/dense/kernel,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE#QNetwork/EncodingNetwork/dense/bias,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEQNetwork/dense_1/kernel,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEQNetwork/dense_1/bias,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE

ref
1

ref
1

_wrapped_policy
 


_q_network
t
_encoder
_q_value_layer
regularization_losses
	variables
trainable_variables
	keras_api
n
_postprocessing_layers
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
	bias
regularization_losses
	variables
trainable_variables
	keras_api
 

0
1
2
	3

0
1
2
	3
�

layers
regularization_losses
 layer_metrics
!layer_regularization_losses
"metrics
#non_trainable_variables
	variables
trainable_variables

$0
%1
 

0
1

0
1
�

&layers
regularization_losses
'layer_metrics
(layer_regularization_losses
)metrics
*non_trainable_variables
	variables
trainable_variables
 

0
	1

0
	1
�

+layers
regularization_losses
,layer_metrics
-layer_regularization_losses
.metrics
/non_trainable_variables
	variables
trainable_variables

0
1
 
 
 
 
R
0regularization_losses
1	variables
2trainable_variables
3	keras_api
h

kernel
bias
4regularization_losses
5	variables
6trainable_variables
7	keras_api

$0
%1
 
 
 
 
 
 
 
 
 
 
 
 
�

8layers
0regularization_losses
9layer_metrics
:layer_regularization_losses
;metrics
<non_trainable_variables
1	variables
2trainable_variables
 

0
1

0
1
�

=layers
4regularization_losses
>layer_metrics
?layer_regularization_losses
@metrics
Anon_trainable_variables
5	variables
6trainable_variables
 
 
 
 
 
 
 
 
 
 
l
action_0/discountPlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
w
action_0/observationPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
j
action_0/rewardPlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
m
action_0/step_typePlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallaction_0/discountaction_0/observationaction_0/rewardaction_0/step_type%QNetwork/EncodingNetwork/dense/kernel#QNetwork/EncodingNetwork/dense/biasQNetwork/dense_1/kernelQNetwork/dense_1/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� */
f*R(
&__inference_signature_wrapper_91200262
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
PartitionedCallPartitionedCallget_initial_state_batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� */
f*R(
&__inference_signature_wrapper_91200274
�
PartitionedCall_1PartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� */
f*R(
&__inference_signature_wrapper_91200296
�
StatefulPartitionedCall_1StatefulPartitionedCallVariable*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� */
f*R(
&__inference_signature_wrapper_91200289
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOp9QNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOp7QNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOp+QNetwork/dense_1/kernel/Read/ReadVariableOp)QNetwork/dense_1/bias/Read/ReadVariableOpConst*
Tin
	2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__traced_save_91200527
�
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariable%QNetwork/EncodingNetwork/dense/kernel#QNetwork/EncodingNetwork/dense/biasQNetwork/dense_1/kernelQNetwork/dense_1/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference__traced_restore_91200552��
�
f
&__inference_signature_wrapper_91200289
unknown:	 
identity	��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *5
f0R.
,__inference_function_with_signature_912002812
StatefulPartitionedCallj
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*
_output_shapes
: 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall
�
8
&__inference_signature_wrapper_91200274

batch_size�
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *5
f0R.
,__inference_function_with_signature_912002692
PartitionedCall*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�
�
&__inference_signature_wrapper_91200262
discount
observation

reward
	step_type
unknown:d
	unknown_0:d
	unknown_1:d
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2*
Tin

2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *5
f0R.
,__inference_function_with_signature_912002442
StatefulPartitionedCallw
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:���������:���������:���������:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:���������
$
_user_specified_name
0/discount:VR
'
_output_shapes
:���������
'
_user_specified_name0/observation:MI
#
_output_shapes
:���������
"
_user_specified_name
0/reward:PL
#
_output_shapes
:���������
%
_user_specified_name0/step_type
�
(
&__inference_signature_wrapper_91200296�
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *5
f0R.
,__inference_function_with_signature_912002922
PartitionedCall*(
_construction_contextkEagerRuntime*
_input_shapes 
�
>
,__inference_function_with_signature_91200269

batch_size�
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� */
f*R(
&__inference_get_initial_state_912002682
PartitionedCall*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�
�
0__inference_polymorphic_distribution_fn_91200481
	step_type

reward
discount
observation��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *$
fR
__inference__raise_912004802
StatefulPartitionedCall*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:���������:���������:���������:���������22
StatefulPartitionedCallStatefulPartitionedCall:N J
#
_output_shapes
:���������
#
_user_specified_name	step_type:KG
#
_output_shapes
:���������
 
_user_specified_namereward:MI
#
_output_shapes
:���������
"
_user_specified_name
discount:TP
'
_output_shapes
:���������
%
_user_specified_nameobservation
�
8
&__inference_get_initial_state_91200484

batch_size*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�
.
,__inference_function_with_signature_91200292�
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *&
f!R
__inference_<lambda>_912000412
PartitionedCall*(
_construction_contextkEagerRuntime*
_input_shapes 
�
d
__inference_<lambda>_91200038!
readvariableop_resource:	 
identity	��ReadVariableOpp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	2
ReadVariableOp`
IdentityIdentityReadVariableOp:value:0^NoOp*
T0	*
_output_shapes
: 2

Identity_
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2 
ReadVariableOpReadVariableOp
�
8
&__inference_get_initial_state_91200268

batch_size*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�n
�
*__inference_polymorphic_action_fn_91200383
	step_type

reward
discount
observationO
=qnetwork_encodingnetwork_dense_matmul_readvariableop_resource:dL
>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource:dA
/qnetwork_dense_1_matmul_readvariableop_resource:d>
0qnetwork_dense_1_biasadd_readvariableop_resource:

identity_1��5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp�4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp�'QNetwork/dense_1/BiasAdd/ReadVariableOp�&QNetwork/dense_1/MatMul/ReadVariableOp�
&QNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2(
&QNetwork/EncodingNetwork/flatten/Const�
(QNetwork/EncodingNetwork/flatten/ReshapeReshapeobservation/QNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:���������2*
(QNetwork/EncodingNetwork/flatten/Reshape�
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp=qnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes

:d*
dtype026
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp�
%QNetwork/EncodingNetwork/dense/MatMulMatMul1QNetwork/EncodingNetwork/flatten/Reshape:output:0<QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2'
%QNetwork/EncodingNetwork/dense/MatMul�
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOp>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype027
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp�
&QNetwork/EncodingNetwork/dense/BiasAddBiasAdd/QNetwork/EncodingNetwork/dense/MatMul:product:0=QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2(
&QNetwork/EncodingNetwork/dense/BiasAdd�
#QNetwork/EncodingNetwork/dense/ReluRelu/QNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������d2%
#QNetwork/EncodingNetwork/dense/Relu�
&QNetwork/dense_1/MatMul/ReadVariableOpReadVariableOp/qnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02(
&QNetwork/dense_1/MatMul/ReadVariableOp�
QNetwork/dense_1/MatMulMatMul1QNetwork/EncodingNetwork/dense/Relu:activations:0.QNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
QNetwork/dense_1/MatMul�
'QNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOp0qnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'QNetwork/dense_1/BiasAdd/ReadVariableOp�
QNetwork/dense_1/BiasAddBiasAdd!QNetwork/dense_1/MatMul:product:0/QNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
QNetwork/dense_1/BiasAdd�
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#Categorical_1/mode/ArgMax/dimension�
Categorical_1/mode/ArgMaxArgMax!QNetwork/dense_1/BiasAdd:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:���������2
Categorical_1/mode/ArgMax�
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtol�
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2%
#Deterministic_1/sample/sample_shape�
Deterministic_1/sample/ShapeShapeCategorical_1/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape~
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_1/sample/Const�
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*Deterministic_1/sample/strided_slice/stack�
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_1�
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2�
$Deterministic_1/sample/strided_sliceStridedSlice%Deterministic_1/sample/Shape:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2&
$Deterministic_1/sample/strided_slice�
'Deterministic_1/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 2)
'Deterministic_1/sample/BroadcastArgs/s0�
)Deterministic_1/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 2+
)Deterministic_1/sample/BroadcastArgs/s0_1�
$Deterministic_1/sample/BroadcastArgsBroadcastArgs2Deterministic_1/sample/BroadcastArgs/s0_1:output:0-Deterministic_1/sample/strided_slice:output:0*
_output_shapes
:2&
$Deterministic_1/sample/BroadcastArgs�
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0�
&Deterministic_1/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB 2(
&Deterministic_1/sample/concat/values_2�
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis�
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0/Deterministic_1/sample/concat/values_2:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat�
"Deterministic_1/sample/BroadcastToBroadcastToCategorical_1/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:���������2$
"Deterministic_1/sample/BroadcastTo�
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_1�
,Deterministic_1/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice_1/stack�
.Deterministic_1/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 20
.Deterministic_1/sample/strided_slice_1/stack_1�
.Deterministic_1/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.Deterministic_1/sample/strided_slice_1/stack_2�
&Deterministic_1/sample/strided_slice_1StridedSlice'Deterministic_1/sample/Shape_1:output:05Deterministic_1/sample/strided_slice_1/stack:output:07Deterministic_1/sample/strided_slice_1/stack_1:output:07Deterministic_1/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2(
&Deterministic_1/sample/strided_slice_1�
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis�
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0/Deterministic_1/sample/strided_slice_1:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1�
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:���������2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :2
clip_by_value/Minimum/y�
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/y�
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_valueG
ShapeShape	step_type*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slicee
shape_as_tensorConst*
_output_shapes
: *
dtype0*
valueB 2
shape_as_tensor\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis�
concatConcatV2strided_slice:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concatj
random_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : 2
random_uniform/minj
random_uniform/maxConst*
_output_shapes
: *
dtype0*
value	B :2
random_uniform/max�
random_uniformRandomUniformIntconcat:output:0random_uniform/min:output:0random_uniform/max:output:0*
T0*

Tout0*#
_output_shapes
:���������2
random_uniform�
IdentityIdentityrandom_uniform:output:0	^discount^observation^reward
^step_type*
T0*#
_output_shapes
:���������2

Identityx
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :2
clip_by_value_1/Minimum/y�
clip_by_value_1/MinimumMinimumIdentity:output:0"clip_by_value_1/Minimum/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value_1/Minimumh
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value_1/y�
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value_1K
Shape_1Shape	step_type*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_1g
epsilon_rng/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
epsilon_rng/ming
epsilon_rng/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
epsilon_rng/max�
epsilon_rng/RandomUniformRandomUniformstrided_slice_1:output:0*
T0*#
_output_shapes
:���������*
dtype02
epsilon_rng/RandomUniform�
epsilon_rng/MulMul"epsilon_rng/RandomUniform:output:0epsilon_rng/max:output:0*
T0*#
_output_shapes
:���������2
epsilon_rng/Mule
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
GreaterEqual/y�
GreaterEqualGreaterEqualepsilon_rng/Mul:z:0GreaterEqual/y:output:0*
T0*#
_output_shapes
:���������2
GreaterEqual�
SelectSelectGreaterEqual:z:0clip_by_value:z:0clip_by_value_1:z:0*
T0*#
_output_shapes
:���������2
Selectx
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :2
clip_by_value_2/Minimum/y�
clip_by_value_2/MinimumMinimumSelect:output:0"clip_by_value_2/Minimum/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value_2/Minimumh
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value_2/y�
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value_2n

Identity_1Identityclip_by_value_2:z:0^NoOp*
T0*#
_output_shapes
:���������2

Identity_1�
NoOpNoOp6^QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp5^QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp(^QNetwork/dense_1/BiasAdd/ReadVariableOp'^QNetwork/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:���������:���������:���������:���������: : : : 2n
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2l
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2R
'QNetwork/dense_1/BiasAdd/ReadVariableOp'QNetwork/dense_1/BiasAdd/ReadVariableOp2P
&QNetwork/dense_1/MatMul/ReadVariableOp&QNetwork/dense_1/MatMul/ReadVariableOp:N J
#
_output_shapes
:���������
#
_user_specified_name	step_type:KG
#
_output_shapes
:���������
 
_user_specified_namereward:MI
#
_output_shapes
:���������
"
_user_specified_name
discount:TP
'
_output_shapes
:���������
%
_user_specified_nameobservation
�
�
!__inference__traced_save_91200527
file_prefix'
#savev2_variable_read_readvariableop	D
@savev2_qnetwork_encodingnetwork_dense_kernel_read_readvariableopB
>savev2_qnetwork_encodingnetwork_dense_bias_read_readvariableop6
2savev2_qnetwork_dense_1_kernel_read_readvariableop4
0savev2_qnetwork_dense_1_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop@savev2_qnetwork_encodingnetwork_dense_kernel_read_readvariableop>savev2_qnetwork_encodingnetwork_dense_bias_read_readvariableop2savev2_qnetwork_dense_1_kernel_read_readvariableop0savev2_qnetwork_dense_1_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes

2	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*9
_input_shapes(
&: : :d:d:d:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :$ 

_output_shapes

:d: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::

_output_shapes
: 
�
l
,__inference_function_with_signature_91200281
unknown:	 
identity	��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *&
f!R
__inference_<lambda>_912000382
StatefulPartitionedCallj
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*
_output_shapes
: 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall
�
�
,__inference_function_with_signature_91200244
	step_type

reward
discount
observation
unknown:d
	unknown_0:d
	unknown_1:d
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2*
Tin

2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *3
f.R,
*__inference_polymorphic_action_fn_912002332
StatefulPartitionedCallw
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:���������:���������:���������:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
#
_output_shapes
:���������
%
_user_specified_name0/step_type:MI
#
_output_shapes
:���������
"
_user_specified_name
0/reward:OK
#
_output_shapes
:���������
$
_user_specified_name
0/discount:VR
'
_output_shapes
:���������
'
_user_specified_name0/observation
�o
�
*__inference_polymorphic_action_fn_91200469
time_step_step_type
time_step_reward
time_step_discount
time_step_observationO
=qnetwork_encodingnetwork_dense_matmul_readvariableop_resource:dL
>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource:dA
/qnetwork_dense_1_matmul_readvariableop_resource:d>
0qnetwork_dense_1_biasadd_readvariableop_resource:

identity_1��5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp�4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp�'QNetwork/dense_1/BiasAdd/ReadVariableOp�&QNetwork/dense_1/MatMul/ReadVariableOp�
&QNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2(
&QNetwork/EncodingNetwork/flatten/Const�
(QNetwork/EncodingNetwork/flatten/ReshapeReshapetime_step_observation/QNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:���������2*
(QNetwork/EncodingNetwork/flatten/Reshape�
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp=qnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes

:d*
dtype026
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp�
%QNetwork/EncodingNetwork/dense/MatMulMatMul1QNetwork/EncodingNetwork/flatten/Reshape:output:0<QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2'
%QNetwork/EncodingNetwork/dense/MatMul�
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOp>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype027
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp�
&QNetwork/EncodingNetwork/dense/BiasAddBiasAdd/QNetwork/EncodingNetwork/dense/MatMul:product:0=QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2(
&QNetwork/EncodingNetwork/dense/BiasAdd�
#QNetwork/EncodingNetwork/dense/ReluRelu/QNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������d2%
#QNetwork/EncodingNetwork/dense/Relu�
&QNetwork/dense_1/MatMul/ReadVariableOpReadVariableOp/qnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02(
&QNetwork/dense_1/MatMul/ReadVariableOp�
QNetwork/dense_1/MatMulMatMul1QNetwork/EncodingNetwork/dense/Relu:activations:0.QNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
QNetwork/dense_1/MatMul�
'QNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOp0qnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'QNetwork/dense_1/BiasAdd/ReadVariableOp�
QNetwork/dense_1/BiasAddBiasAdd!QNetwork/dense_1/MatMul:product:0/QNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
QNetwork/dense_1/BiasAdd�
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#Categorical_1/mode/ArgMax/dimension�
Categorical_1/mode/ArgMaxArgMax!QNetwork/dense_1/BiasAdd:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:���������2
Categorical_1/mode/ArgMax�
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtol�
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2%
#Deterministic_1/sample/sample_shape�
Deterministic_1/sample/ShapeShapeCategorical_1/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape~
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_1/sample/Const�
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*Deterministic_1/sample/strided_slice/stack�
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_1�
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2�
$Deterministic_1/sample/strided_sliceStridedSlice%Deterministic_1/sample/Shape:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2&
$Deterministic_1/sample/strided_slice�
'Deterministic_1/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 2)
'Deterministic_1/sample/BroadcastArgs/s0�
)Deterministic_1/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 2+
)Deterministic_1/sample/BroadcastArgs/s0_1�
$Deterministic_1/sample/BroadcastArgsBroadcastArgs2Deterministic_1/sample/BroadcastArgs/s0_1:output:0-Deterministic_1/sample/strided_slice:output:0*
_output_shapes
:2&
$Deterministic_1/sample/BroadcastArgs�
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0�
&Deterministic_1/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB 2(
&Deterministic_1/sample/concat/values_2�
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis�
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0/Deterministic_1/sample/concat/values_2:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat�
"Deterministic_1/sample/BroadcastToBroadcastToCategorical_1/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:���������2$
"Deterministic_1/sample/BroadcastTo�
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_1�
,Deterministic_1/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice_1/stack�
.Deterministic_1/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 20
.Deterministic_1/sample/strided_slice_1/stack_1�
.Deterministic_1/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.Deterministic_1/sample/strided_slice_1/stack_2�
&Deterministic_1/sample/strided_slice_1StridedSlice'Deterministic_1/sample/Shape_1:output:05Deterministic_1/sample/strided_slice_1/stack:output:07Deterministic_1/sample/strided_slice_1/stack_1:output:07Deterministic_1/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2(
&Deterministic_1/sample/strided_slice_1�
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis�
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0/Deterministic_1/sample/strided_slice_1:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1�
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:���������2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :2
clip_by_value/Minimum/y�
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/y�
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_valueQ
ShapeShapetime_step_step_type*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slicee
shape_as_tensorConst*
_output_shapes
: *
dtype0*
valueB 2
shape_as_tensor\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis�
concatConcatV2strided_slice:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concatj
random_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : 2
random_uniform/minj
random_uniform/maxConst*
_output_shapes
: *
dtype0*
value	B :2
random_uniform/max�
random_uniformRandomUniformIntconcat:output:0random_uniform/min:output:0random_uniform/max:output:0*
T0*

Tout0*#
_output_shapes
:���������2
random_uniform�
IdentityIdentityrandom_uniform:output:0^time_step_discount^time_step_observation^time_step_reward^time_step_step_type*
T0*#
_output_shapes
:���������2

Identityx
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :2
clip_by_value_1/Minimum/y�
clip_by_value_1/MinimumMinimumIdentity:output:0"clip_by_value_1/Minimum/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value_1/Minimumh
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value_1/y�
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value_1U
Shape_1Shapetime_step_step_type*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_1g
epsilon_rng/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
epsilon_rng/ming
epsilon_rng/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
epsilon_rng/max�
epsilon_rng/RandomUniformRandomUniformstrided_slice_1:output:0*
T0*#
_output_shapes
:���������*
dtype02
epsilon_rng/RandomUniform�
epsilon_rng/MulMul"epsilon_rng/RandomUniform:output:0epsilon_rng/max:output:0*
T0*#
_output_shapes
:���������2
epsilon_rng/Mule
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
GreaterEqual/y�
GreaterEqualGreaterEqualepsilon_rng/Mul:z:0GreaterEqual/y:output:0*
T0*#
_output_shapes
:���������2
GreaterEqual�
SelectSelectGreaterEqual:z:0clip_by_value:z:0clip_by_value_1:z:0*
T0*#
_output_shapes
:���������2
Selectx
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :2
clip_by_value_2/Minimum/y�
clip_by_value_2/MinimumMinimumSelect:output:0"clip_by_value_2/Minimum/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value_2/Minimumh
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value_2/y�
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value_2n

Identity_1Identityclip_by_value_2:z:0^NoOp*
T0*#
_output_shapes
:���������2

Identity_1�
NoOpNoOp6^QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp5^QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp(^QNetwork/dense_1/BiasAdd/ReadVariableOp'^QNetwork/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:���������:���������:���������:���������: : : : 2n
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2l
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2R
'QNetwork/dense_1/BiasAdd/ReadVariableOp'QNetwork/dense_1/BiasAdd/ReadVariableOp2P
&QNetwork/dense_1/MatMul/ReadVariableOp&QNetwork/dense_1/MatMul/ReadVariableOp:X T
#
_output_shapes
:���������
-
_user_specified_nametime_step/step_type:UQ
#
_output_shapes
:���������
*
_user_specified_nametime_step/reward:WS
#
_output_shapes
:���������
,
_user_specified_nametime_step/discount:^Z
'
_output_shapes
:���������
/
_user_specified_nametime_step/observation
^

__inference_<lambda>_91200041*(
_construction_contextkEagerRuntime*
_input_shapes 
�n
�
*__inference_polymorphic_action_fn_91200233
	time_step
time_step_1
time_step_2
time_step_3O
=qnetwork_encodingnetwork_dense_matmul_readvariableop_resource:dL
>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource:dA
/qnetwork_dense_1_matmul_readvariableop_resource:d>
0qnetwork_dense_1_biasadd_readvariableop_resource:

identity_1��5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp�4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp�'QNetwork/dense_1/BiasAdd/ReadVariableOp�&QNetwork/dense_1/MatMul/ReadVariableOp�
&QNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2(
&QNetwork/EncodingNetwork/flatten/Const�
(QNetwork/EncodingNetwork/flatten/ReshapeReshapetime_step_3/QNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:���������2*
(QNetwork/EncodingNetwork/flatten/Reshape�
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp=qnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes

:d*
dtype026
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp�
%QNetwork/EncodingNetwork/dense/MatMulMatMul1QNetwork/EncodingNetwork/flatten/Reshape:output:0<QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2'
%QNetwork/EncodingNetwork/dense/MatMul�
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOp>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype027
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp�
&QNetwork/EncodingNetwork/dense/BiasAddBiasAdd/QNetwork/EncodingNetwork/dense/MatMul:product:0=QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2(
&QNetwork/EncodingNetwork/dense/BiasAdd�
#QNetwork/EncodingNetwork/dense/ReluRelu/QNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������d2%
#QNetwork/EncodingNetwork/dense/Relu�
&QNetwork/dense_1/MatMul/ReadVariableOpReadVariableOp/qnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02(
&QNetwork/dense_1/MatMul/ReadVariableOp�
QNetwork/dense_1/MatMulMatMul1QNetwork/EncodingNetwork/dense/Relu:activations:0.QNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
QNetwork/dense_1/MatMul�
'QNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOp0qnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'QNetwork/dense_1/BiasAdd/ReadVariableOp�
QNetwork/dense_1/BiasAddBiasAdd!QNetwork/dense_1/MatMul:product:0/QNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
QNetwork/dense_1/BiasAdd�
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#Categorical_1/mode/ArgMax/dimension�
Categorical_1/mode/ArgMaxArgMax!QNetwork/dense_1/BiasAdd:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:���������2
Categorical_1/mode/ArgMax�
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtol�
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2%
#Deterministic_1/sample/sample_shape�
Deterministic_1/sample/ShapeShapeCategorical_1/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape~
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_1/sample/Const�
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*Deterministic_1/sample/strided_slice/stack�
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_1�
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2�
$Deterministic_1/sample/strided_sliceStridedSlice%Deterministic_1/sample/Shape:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2&
$Deterministic_1/sample/strided_slice�
'Deterministic_1/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 2)
'Deterministic_1/sample/BroadcastArgs/s0�
)Deterministic_1/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 2+
)Deterministic_1/sample/BroadcastArgs/s0_1�
$Deterministic_1/sample/BroadcastArgsBroadcastArgs2Deterministic_1/sample/BroadcastArgs/s0_1:output:0-Deterministic_1/sample/strided_slice:output:0*
_output_shapes
:2&
$Deterministic_1/sample/BroadcastArgs�
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0�
&Deterministic_1/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB 2(
&Deterministic_1/sample/concat/values_2�
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis�
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0/Deterministic_1/sample/concat/values_2:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat�
"Deterministic_1/sample/BroadcastToBroadcastToCategorical_1/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:���������2$
"Deterministic_1/sample/BroadcastTo�
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_1�
,Deterministic_1/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice_1/stack�
.Deterministic_1/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 20
.Deterministic_1/sample/strided_slice_1/stack_1�
.Deterministic_1/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.Deterministic_1/sample/strided_slice_1/stack_2�
&Deterministic_1/sample/strided_slice_1StridedSlice'Deterministic_1/sample/Shape_1:output:05Deterministic_1/sample/strided_slice_1/stack:output:07Deterministic_1/sample/strided_slice_1/stack_1:output:07Deterministic_1/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2(
&Deterministic_1/sample/strided_slice_1�
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis�
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0/Deterministic_1/sample/strided_slice_1:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1�
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:���������2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :2
clip_by_value/Minimum/y�
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/y�
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_valueG
ShapeShape	time_step*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slicee
shape_as_tensorConst*
_output_shapes
: *
dtype0*
valueB 2
shape_as_tensor\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis�
concatConcatV2strided_slice:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concatj
random_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : 2
random_uniform/minj
random_uniform/maxConst*
_output_shapes
: *
dtype0*
value	B :2
random_uniform/max�
random_uniformRandomUniformIntconcat:output:0random_uniform/min:output:0random_uniform/max:output:0*
T0*

Tout0*#
_output_shapes
:���������2
random_uniform�
IdentityIdentityrandom_uniform:output:0
^time_step^time_step_1^time_step_2^time_step_3*
T0*#
_output_shapes
:���������2

Identityx
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :2
clip_by_value_1/Minimum/y�
clip_by_value_1/MinimumMinimumIdentity:output:0"clip_by_value_1/Minimum/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value_1/Minimumh
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value_1/y�
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value_1K
Shape_1Shape	time_step*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_1g
epsilon_rng/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
epsilon_rng/ming
epsilon_rng/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
epsilon_rng/max�
epsilon_rng/RandomUniformRandomUniformstrided_slice_1:output:0*
T0*#
_output_shapes
:���������*
dtype02
epsilon_rng/RandomUniform�
epsilon_rng/MulMul"epsilon_rng/RandomUniform:output:0epsilon_rng/max:output:0*
T0*#
_output_shapes
:���������2
epsilon_rng/Mule
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
GreaterEqual/y�
GreaterEqualGreaterEqualepsilon_rng/Mul:z:0GreaterEqual/y:output:0*
T0*#
_output_shapes
:���������2
GreaterEqual�
SelectSelectGreaterEqual:z:0clip_by_value:z:0clip_by_value_1:z:0*
T0*#
_output_shapes
:���������2
Selectx
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :2
clip_by_value_2/Minimum/y�
clip_by_value_2/MinimumMinimumSelect:output:0"clip_by_value_2/Minimum/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value_2/Minimumh
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value_2/y�
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value_2n

Identity_1Identityclip_by_value_2:z:0^NoOp*
T0*#
_output_shapes
:���������2

Identity_1�
NoOpNoOp6^QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp5^QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp(^QNetwork/dense_1/BiasAdd/ReadVariableOp'^QNetwork/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:���������:���������:���������:���������: : : : 2n
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2l
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2R
'QNetwork/dense_1/BiasAdd/ReadVariableOp'QNetwork/dense_1/BiasAdd/ReadVariableOp2P
&QNetwork/dense_1/MatMul/ReadVariableOp&QNetwork/dense_1/MatMul/ReadVariableOp:N J
#
_output_shapes
:���������
#
_user_specified_name	time_step:NJ
#
_output_shapes
:���������
#
_user_specified_name	time_step:NJ
#
_output_shapes
:���������
#
_user_specified_name	time_step:RN
'
_output_shapes
:���������
#
_user_specified_name	time_step
�
�
$__inference__traced_restore_91200552
file_prefix#
assignvariableop_variable:	 J
8assignvariableop_1_qnetwork_encodingnetwork_dense_kernel:dD
6assignvariableop_2_qnetwork_encodingnetwork_dense_bias:d<
*assignvariableop_3_qnetwork_dense_1_kernel:d6
(assignvariableop_4_qnetwork_dense_1_bias:

identity_6��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*,
_output_shapes
::::::*
dtypes

2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp8assignvariableop_1_qnetwork_encodingnetwork_dense_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp6assignvariableop_2_qnetwork_encodingnetwork_dense_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp*assignvariableop_3_qnetwork_dense_1_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp(assignvariableop_4_qnetwork_dense_1_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�

Identity_5Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_5c

Identity_6IdentityIdentity_5:output:0^NoOp_1*
T0*
_output_shapes
: 2

Identity_6�
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"!

identity_6Identity_6:output:0*
_input_shapes
: : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_4:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
0
__inference__raise_91200480��Assert/Assert�
Assert/ConstConst*
_output_shapes
: *
dtype0*H
value?B= B7EpsilonGreedyPolicy does not support distributions yet.2
Assert/Constt
Assert/Assert/conditionConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
Assert/Assert/condition�
Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*H
value?B= B7EpsilonGreedyPolicy does not support distributions yet.2
Assert/Assert/data_0�
Assert/AssertAssert Assert/Assert/condition:output:0Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2
Assert/Assert*(
_construction_contextkEagerRuntime*
_input_shapes 2
Assert/AssertAssert/Assert"�L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
action�
4

0/discount&
action_0/discount:0���������
>
0/observation-
action_0/observation:0���������
0
0/reward$
action_0/reward:0���������
6
0/step_type'
action_0/step_type:0���������6
action,
StatefulPartitionedCall:0���������tensorflow/serving/predict*e
get_initial_stateP
2

batch_size$
get_initial_state_batch_size:0 tensorflow/serving/predict*,
get_metadatatensorflow/serving/predict*Z
get_train_stepH*
int64!
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:�S
�

train_step
metadata
model_variables
_all_assets

signatures

Baction
Cdistribution
Dget_initial_state
Eget_metadata
Fget_train_step"
_generic_user_object
:	 (2Variable
 "
trackable_dict_wrapper
=
0
1
2
	3"
trackable_tuple_wrapper
.

0
1"
trackable_list_wrapper
`

Gaction
Hget_initial_state
Iget_train_step
Jget_metadata"
signature_map
7:5d2%QNetwork/EncodingNetwork/dense/kernel
1:/d2#QNetwork/EncodingNetwork/dense/bias
):'d2QNetwork/dense_1/kernel
#:!2QNetwork/dense_1/bias
1
ref
1"
trackable_tuple_wrapper
1
ref
1"
trackable_tuple_wrapper
3
_wrapped_policy"
_generic_user_object
"
_generic_user_object
.

_q_network"
_generic_user_object
�
_encoder
_q_value_layer
regularization_losses
	variables
trainable_variables
	keras_api
K__call__
*L&call_and_return_all_conditional_losses"
_tf_keras_layer
�
_postprocessing_layers
regularization_losses
	variables
trainable_variables
	keras_api
M__call__
*N&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
	bias
regularization_losses
	variables
trainable_variables
	keras_api
O__call__
*P&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
<
0
1
2
	3"
trackable_list_wrapper
<
0
1
2
	3"
trackable_list_wrapper
�

layers
regularization_losses
 layer_metrics
!layer_regularization_losses
"metrics
#non_trainable_variables
	variables
trainable_variables
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�

&layers
regularization_losses
'layer_metrics
(layer_regularization_losses
)metrics
*non_trainable_variables
	variables
trainable_variables
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
�

+layers
regularization_losses
,layer_metrics
-layer_regularization_losses
.metrics
/non_trainable_variables
	variables
trainable_variables
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
0regularization_losses
1	variables
2trainable_variables
3	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
bias
4regularization_losses
5	variables
6trainable_variables
7	keras_api
S__call__
*T&call_and_return_all_conditional_losses"
_tf_keras_layer
.
$0
%1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

8layers
0regularization_losses
9layer_metrics
:layer_regularization_losses
;metrics
<non_trainable_variables
1	variables
2trainable_variables
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�

=layers
4regularization_losses
>layer_metrics
?layer_regularization_losses
@metrics
Anon_trainable_variables
5	variables
6trainable_variables
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
*__inference_polymorphic_action_fn_91200383
*__inference_polymorphic_action_fn_91200469�
���
FullArgSpec(
args �
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults�
� 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
0__inference_polymorphic_distribution_fn_91200481�
���
FullArgSpec(
args �
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults�
� 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_get_initial_state_91200484�
���
FullArgSpec!
args�
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_<lambda>_91200041"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_<lambda>_91200038"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_signature_wrapper_91200262
0/discount0/observation0/reward0/step_type"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_signature_wrapper_91200274
batch_size"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_signature_wrapper_91200289"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_signature_wrapper_91200296"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpecL
argsD�A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpecL
argsD�A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpecL
argsD�A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpecL
argsD�A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 <
__inference_<lambda>_91200038�

� 
� "� 	5
__inference_<lambda>_91200041�

� 
� "� S
&__inference_get_initial_state_91200484)"�
�
�

batch_size 
� "� �
*__inference_polymorphic_action_fn_91200383�	���
���
���
TimeStep,
	step_type�
	step_type���������&
reward�
reward���������*
discount�
discount���������4
observation%�"
observation���������
� 
� "R�O

PolicyStep&
action�
action���������
state� 
info� �
*__inference_polymorphic_action_fn_91200469�	���
���
���
TimeStep6
	step_type)�&
time_step/step_type���������0
reward&�#
time_step/reward���������4
discount(�%
time_step/discount���������>
observation/�,
time_step/observation���������
� 
� "R�O

PolicyStep&
action�
action���������
state� 
info� �
0__inference_polymorphic_distribution_fn_91200481����
���
���
TimeStep,
	step_type�
	step_type���������&
reward�
reward���������*
discount�
discount���������4
observation%�"
observation���������
� 
� "
 �
&__inference_signature_wrapper_91200262�	���
� 
���
.

0/discount �

0/discount���������
8
0/observation'�$
0/observation���������
*
0/reward�
0/reward���������
0
0/step_type!�
0/step_type���������"+�(
&
action�
action���������a
&__inference_signature_wrapper_9120027470�-
� 
&�#
!

batch_size�

batch_size "� Z
&__inference_signature_wrapper_912002890�

� 
� "�

int64�
int64 	>
&__inference_signature_wrapper_91200296�

� 
� "� 