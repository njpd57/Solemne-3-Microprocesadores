��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
�
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	"
grad_abool( "
grad_bbool( 
�
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
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
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
f
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx" 
Tidxtype0:
2
	
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
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
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
executor_typestring ��
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
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListReserve
element_shape"
shape_type
num_elements(
handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint���������
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
9
VarIsInitializedOp
resource
is_initialized
�
�
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
�"serve*2.18.02v2.18.0-rc2-4-g6550e4bd8028��
�
lstm/lstm_cell/biasVarHandleOp*
_output_shapes
: *$

debug_namelstm/lstm_cell/bias/*
dtype0*
shape:P*$
shared_namelstm/lstm_cell/bias
w
'lstm/lstm_cell/bias/Read/ReadVariableOpReadVariableOplstm/lstm_cell/bias*
_output_shapes
:P*
dtype0
�
#Variable/Initializer/ReadVariableOpReadVariableOplstm/lstm_cell/bias*
_class
loc:@Variable*
_output_shapes
:P*
dtype0
�
VariableVarHandleOp*
_class
loc:@Variable*
_output_shapes
: *

debug_name	Variable/*
dtype0*
shape:P*
shared_name
Variable
a
)Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable*
_output_shapes
: 
_
Variable/AssignAssignVariableOpVariable#Variable/Initializer/ReadVariableOp*
dtype0
a
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:P*
dtype0
�
lstm/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *0

debug_name" lstm/lstm_cell/recurrent_kernel/*
dtype0*
shape
:P*0
shared_name!lstm/lstm_cell/recurrent_kernel
�
3lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOplstm/lstm_cell/recurrent_kernel*
_output_shapes

:P*
dtype0
�
%Variable_1/Initializer/ReadVariableOpReadVariableOplstm/lstm_cell/recurrent_kernel*
_class
loc:@Variable_1*
_output_shapes

:P*
dtype0
�

Variable_1VarHandleOp*
_class
loc:@Variable_1*
_output_shapes
: *

debug_nameVariable_1/*
dtype0*
shape
:P*
shared_name
Variable_1
e
+Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_1*
_output_shapes
: 
e
Variable_1/AssignAssignVariableOp
Variable_1%Variable_1/Initializer/ReadVariableOp*
dtype0
i
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes

:P*
dtype0
�
lstm/lstm_cell/kernelVarHandleOp*
_output_shapes
: *&

debug_namelstm/lstm_cell/kernel/*
dtype0*
shape
:P*&
shared_namelstm/lstm_cell/kernel

)lstm/lstm_cell/kernel/Read/ReadVariableOpReadVariableOplstm/lstm_cell/kernel*
_output_shapes

:P*
dtype0
�
%Variable_2/Initializer/ReadVariableOpReadVariableOplstm/lstm_cell/kernel*
_class
loc:@Variable_2*
_output_shapes

:P*
dtype0
�

Variable_2VarHandleOp*
_class
loc:@Variable_2*
_output_shapes
: *

debug_nameVariable_2/*
dtype0*
shape
:P*
shared_name
Variable_2
e
+Variable_2/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_2*
_output_shapes
: 
e
Variable_2/AssignAssignVariableOp
Variable_2%Variable_2/Initializer/ReadVariableOp*
dtype0
i
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes

:P*
dtype0
�
adam/dense_bias_velocityVarHandleOp*
_output_shapes
: *)

debug_nameadam/dense_bias_velocity/*
dtype0*
shape:
*)
shared_nameadam/dense_bias_velocity
�
,adam/dense_bias_velocity/Read/ReadVariableOpReadVariableOpadam/dense_bias_velocity*
_output_shapes
:
*
dtype0
�
%Variable_3/Initializer/ReadVariableOpReadVariableOpadam/dense_bias_velocity*
_class
loc:@Variable_3*
_output_shapes
:
*
dtype0
�

Variable_3VarHandleOp*
_class
loc:@Variable_3*
_output_shapes
: *

debug_nameVariable_3/*
dtype0*
shape:
*
shared_name
Variable_3
e
+Variable_3/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_3*
_output_shapes
: 
e
Variable_3/AssignAssignVariableOp
Variable_3%Variable_3/Initializer/ReadVariableOp*
dtype0
e
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*
_output_shapes
:
*
dtype0
�
adam/dense_bias_momentumVarHandleOp*
_output_shapes
: *)

debug_nameadam/dense_bias_momentum/*
dtype0*
shape:
*)
shared_nameadam/dense_bias_momentum
�
,adam/dense_bias_momentum/Read/ReadVariableOpReadVariableOpadam/dense_bias_momentum*
_output_shapes
:
*
dtype0
�
%Variable_4/Initializer/ReadVariableOpReadVariableOpadam/dense_bias_momentum*
_class
loc:@Variable_4*
_output_shapes
:
*
dtype0
�

Variable_4VarHandleOp*
_class
loc:@Variable_4*
_output_shapes
: *

debug_nameVariable_4/*
dtype0*
shape:
*
shared_name
Variable_4
e
+Variable_4/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_4*
_output_shapes
: 
e
Variable_4/AssignAssignVariableOp
Variable_4%Variable_4/Initializer/ReadVariableOp*
dtype0
e
Variable_4/Read/ReadVariableOpReadVariableOp
Variable_4*
_output_shapes
:
*
dtype0
�
adam/dense_kernel_velocityVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_kernel_velocity/*
dtype0*
shape
:
*+
shared_nameadam/dense_kernel_velocity
�
.adam/dense_kernel_velocity/Read/ReadVariableOpReadVariableOpadam/dense_kernel_velocity*
_output_shapes

:
*
dtype0
�
%Variable_5/Initializer/ReadVariableOpReadVariableOpadam/dense_kernel_velocity*
_class
loc:@Variable_5*
_output_shapes

:
*
dtype0
�

Variable_5VarHandleOp*
_class
loc:@Variable_5*
_output_shapes
: *

debug_nameVariable_5/*
dtype0*
shape
:
*
shared_name
Variable_5
e
+Variable_5/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_5*
_output_shapes
: 
e
Variable_5/AssignAssignVariableOp
Variable_5%Variable_5/Initializer/ReadVariableOp*
dtype0
i
Variable_5/Read/ReadVariableOpReadVariableOp
Variable_5*
_output_shapes

:
*
dtype0
�
adam/dense_kernel_momentumVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_kernel_momentum/*
dtype0*
shape
:
*+
shared_nameadam/dense_kernel_momentum
�
.adam/dense_kernel_momentum/Read/ReadVariableOpReadVariableOpadam/dense_kernel_momentum*
_output_shapes

:
*
dtype0
�
%Variable_6/Initializer/ReadVariableOpReadVariableOpadam/dense_kernel_momentum*
_class
loc:@Variable_6*
_output_shapes

:
*
dtype0
�

Variable_6VarHandleOp*
_class
loc:@Variable_6*
_output_shapes
: *

debug_nameVariable_6/*
dtype0*
shape
:
*
shared_name
Variable_6
e
+Variable_6/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_6*
_output_shapes
: 
e
Variable_6/AssignAssignVariableOp
Variable_6%Variable_6/Initializer/ReadVariableOp*
dtype0
i
Variable_6/Read/ReadVariableOpReadVariableOp
Variable_6*
_output_shapes

:
*
dtype0
�
!adam/lstm_lstm_cell_bias_velocityVarHandleOp*
_output_shapes
: *2

debug_name$"adam/lstm_lstm_cell_bias_velocity/*
dtype0*
shape:P*2
shared_name#!adam/lstm_lstm_cell_bias_velocity
�
5adam/lstm_lstm_cell_bias_velocity/Read/ReadVariableOpReadVariableOp!adam/lstm_lstm_cell_bias_velocity*
_output_shapes
:P*
dtype0
�
%Variable_7/Initializer/ReadVariableOpReadVariableOp!adam/lstm_lstm_cell_bias_velocity*
_class
loc:@Variable_7*
_output_shapes
:P*
dtype0
�

Variable_7VarHandleOp*
_class
loc:@Variable_7*
_output_shapes
: *

debug_nameVariable_7/*
dtype0*
shape:P*
shared_name
Variable_7
e
+Variable_7/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_7*
_output_shapes
: 
e
Variable_7/AssignAssignVariableOp
Variable_7%Variable_7/Initializer/ReadVariableOp*
dtype0
e
Variable_7/Read/ReadVariableOpReadVariableOp
Variable_7*
_output_shapes
:P*
dtype0
�
!adam/lstm_lstm_cell_bias_momentumVarHandleOp*
_output_shapes
: *2

debug_name$"adam/lstm_lstm_cell_bias_momentum/*
dtype0*
shape:P*2
shared_name#!adam/lstm_lstm_cell_bias_momentum
�
5adam/lstm_lstm_cell_bias_momentum/Read/ReadVariableOpReadVariableOp!adam/lstm_lstm_cell_bias_momentum*
_output_shapes
:P*
dtype0
�
%Variable_8/Initializer/ReadVariableOpReadVariableOp!adam/lstm_lstm_cell_bias_momentum*
_class
loc:@Variable_8*
_output_shapes
:P*
dtype0
�

Variable_8VarHandleOp*
_class
loc:@Variable_8*
_output_shapes
: *

debug_nameVariable_8/*
dtype0*
shape:P*
shared_name
Variable_8
e
+Variable_8/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_8*
_output_shapes
: 
e
Variable_8/AssignAssignVariableOp
Variable_8%Variable_8/Initializer/ReadVariableOp*
dtype0
e
Variable_8/Read/ReadVariableOpReadVariableOp
Variable_8*
_output_shapes
:P*
dtype0
�
-adam/lstm_lstm_cell_recurrent_kernel_velocityVarHandleOp*
_output_shapes
: *>

debug_name0.adam/lstm_lstm_cell_recurrent_kernel_velocity/*
dtype0*
shape
:P*>
shared_name/-adam/lstm_lstm_cell_recurrent_kernel_velocity
�
Aadam/lstm_lstm_cell_recurrent_kernel_velocity/Read/ReadVariableOpReadVariableOp-adam/lstm_lstm_cell_recurrent_kernel_velocity*
_output_shapes

:P*
dtype0
�
%Variable_9/Initializer/ReadVariableOpReadVariableOp-adam/lstm_lstm_cell_recurrent_kernel_velocity*
_class
loc:@Variable_9*
_output_shapes

:P*
dtype0
�

Variable_9VarHandleOp*
_class
loc:@Variable_9*
_output_shapes
: *

debug_nameVariable_9/*
dtype0*
shape
:P*
shared_name
Variable_9
e
+Variable_9/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_9*
_output_shapes
: 
e
Variable_9/AssignAssignVariableOp
Variable_9%Variable_9/Initializer/ReadVariableOp*
dtype0
i
Variable_9/Read/ReadVariableOpReadVariableOp
Variable_9*
_output_shapes

:P*
dtype0
�
-adam/lstm_lstm_cell_recurrent_kernel_momentumVarHandleOp*
_output_shapes
: *>

debug_name0.adam/lstm_lstm_cell_recurrent_kernel_momentum/*
dtype0*
shape
:P*>
shared_name/-adam/lstm_lstm_cell_recurrent_kernel_momentum
�
Aadam/lstm_lstm_cell_recurrent_kernel_momentum/Read/ReadVariableOpReadVariableOp-adam/lstm_lstm_cell_recurrent_kernel_momentum*
_output_shapes

:P*
dtype0
�
&Variable_10/Initializer/ReadVariableOpReadVariableOp-adam/lstm_lstm_cell_recurrent_kernel_momentum*
_class
loc:@Variable_10*
_output_shapes

:P*
dtype0
�
Variable_10VarHandleOp*
_class
loc:@Variable_10*
_output_shapes
: *

debug_nameVariable_10/*
dtype0*
shape
:P*
shared_nameVariable_10
g
,Variable_10/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_10*
_output_shapes
: 
h
Variable_10/AssignAssignVariableOpVariable_10&Variable_10/Initializer/ReadVariableOp*
dtype0
k
Variable_10/Read/ReadVariableOpReadVariableOpVariable_10*
_output_shapes

:P*
dtype0
�
#adam/lstm_lstm_cell_kernel_velocityVarHandleOp*
_output_shapes
: *4

debug_name&$adam/lstm_lstm_cell_kernel_velocity/*
dtype0*
shape
:P*4
shared_name%#adam/lstm_lstm_cell_kernel_velocity
�
7adam/lstm_lstm_cell_kernel_velocity/Read/ReadVariableOpReadVariableOp#adam/lstm_lstm_cell_kernel_velocity*
_output_shapes

:P*
dtype0
�
&Variable_11/Initializer/ReadVariableOpReadVariableOp#adam/lstm_lstm_cell_kernel_velocity*
_class
loc:@Variable_11*
_output_shapes

:P*
dtype0
�
Variable_11VarHandleOp*
_class
loc:@Variable_11*
_output_shapes
: *

debug_nameVariable_11/*
dtype0*
shape
:P*
shared_nameVariable_11
g
,Variable_11/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_11*
_output_shapes
: 
h
Variable_11/AssignAssignVariableOpVariable_11&Variable_11/Initializer/ReadVariableOp*
dtype0
k
Variable_11/Read/ReadVariableOpReadVariableOpVariable_11*
_output_shapes

:P*
dtype0
�
#adam/lstm_lstm_cell_kernel_momentumVarHandleOp*
_output_shapes
: *4

debug_name&$adam/lstm_lstm_cell_kernel_momentum/*
dtype0*
shape
:P*4
shared_name%#adam/lstm_lstm_cell_kernel_momentum
�
7adam/lstm_lstm_cell_kernel_momentum/Read/ReadVariableOpReadVariableOp#adam/lstm_lstm_cell_kernel_momentum*
_output_shapes

:P*
dtype0
�
&Variable_12/Initializer/ReadVariableOpReadVariableOp#adam/lstm_lstm_cell_kernel_momentum*
_class
loc:@Variable_12*
_output_shapes

:P*
dtype0
�
Variable_12VarHandleOp*
_class
loc:@Variable_12*
_output_shapes
: *

debug_nameVariable_12/*
dtype0*
shape
:P*
shared_nameVariable_12
g
,Variable_12/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_12*
_output_shapes
: 
h
Variable_12/AssignAssignVariableOpVariable_12&Variable_12/Initializer/ReadVariableOp*
dtype0
k
Variable_12/Read/ReadVariableOpReadVariableOpVariable_12*
_output_shapes

:P*
dtype0
�

dense/biasVarHandleOp*
_output_shapes
: *

debug_namedense/bias/*
dtype0*
shape:
*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:
*
dtype0
�
&Variable_13/Initializer/ReadVariableOpReadVariableOp
dense/bias*
_class
loc:@Variable_13*
_output_shapes
:
*
dtype0
�
Variable_13VarHandleOp*
_class
loc:@Variable_13*
_output_shapes
: *

debug_nameVariable_13/*
dtype0*
shape:
*
shared_nameVariable_13
g
,Variable_13/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_13*
_output_shapes
: 
h
Variable_13/AssignAssignVariableOpVariable_13&Variable_13/Initializer/ReadVariableOp*
dtype0
g
Variable_13/Read/ReadVariableOpReadVariableOpVariable_13*
_output_shapes
:
*
dtype0
�
dense/kernelVarHandleOp*
_output_shapes
: *

debug_namedense/kernel/*
dtype0*
shape
:
*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:
*
dtype0
�
&Variable_14/Initializer/ReadVariableOpReadVariableOpdense/kernel*
_class
loc:@Variable_14*
_output_shapes

:
*
dtype0
�
Variable_14VarHandleOp*
_class
loc:@Variable_14*
_output_shapes
: *

debug_nameVariable_14/*
dtype0*
shape
:
*
shared_nameVariable_14
g
,Variable_14/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_14*
_output_shapes
: 
h
Variable_14/AssignAssignVariableOpVariable_14&Variable_14/Initializer/ReadVariableOp*
dtype0
k
Variable_14/Read/ReadVariableOpReadVariableOpVariable_14*
_output_shapes

:
*
dtype0
�
adam/learning_rateVarHandleOp*
_output_shapes
: *#

debug_nameadam/learning_rate/*
dtype0*
shape: *#
shared_nameadam/learning_rate
q
&adam/learning_rate/Read/ReadVariableOpReadVariableOpadam/learning_rate*
_output_shapes
: *
dtype0
�
&Variable_15/Initializer/ReadVariableOpReadVariableOpadam/learning_rate*
_class
loc:@Variable_15*
_output_shapes
: *
dtype0
�
Variable_15VarHandleOp*
_class
loc:@Variable_15*
_output_shapes
: *

debug_nameVariable_15/*
dtype0*
shape: *
shared_nameVariable_15
g
,Variable_15/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_15*
_output_shapes
: 
h
Variable_15/AssignAssignVariableOpVariable_15&Variable_15/Initializer/ReadVariableOp*
dtype0
c
Variable_15/Read/ReadVariableOpReadVariableOpVariable_15*
_output_shapes
: *
dtype0
�
adam/iterationVarHandleOp*
_output_shapes
: *

debug_nameadam/iteration/*
dtype0	*
shape: *
shared_nameadam/iteration
i
"adam/iteration/Read/ReadVariableOpReadVariableOpadam/iteration*
_output_shapes
: *
dtype0	
�
&Variable_16/Initializer/ReadVariableOpReadVariableOpadam/iteration*
_class
loc:@Variable_16*
_output_shapes
: *
dtype0	
�
Variable_16VarHandleOp*
_class
loc:@Variable_16*
_output_shapes
: *

debug_nameVariable_16/*
dtype0	*
shape: *
shared_nameVariable_16
g
,Variable_16/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_16*
_output_shapes
: 
h
Variable_16/AssignAssignVariableOpVariable_16&Variable_16/Initializer/ReadVariableOp*
dtype0	
c
Variable_16/Read/ReadVariableOpReadVariableOpVariable_16*
_output_shapes
: *
dtype0	
o
serving_default_inputsPlaceholder*"
_output_shapes
:*
dtype0*
shape:
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputslstm/lstm_cell/kernellstm/lstm_cell/recurrent_kernellstm/lstm_cell/biasdense/kernel
dense/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*'
_read_only_resource_inputs	
*2
config_proto" 

CPU

GPU 2J 8� �J *<
f7R5
3__inference_signature_wrapper_serving_default_37158

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
_tracked
_inbound_nodes
_outbound_nodes
_losses
_losses_override
_operations
_layers
_build_shapes_dict
	output_names

	optimizer
_default_save_signature

signatures*
* 
* 
* 
* 
* 
 
0
1
2
3*
 
0
1
2
3*
* 
* 
�

_variables
_trainable_variables
 _trainable_variables_indices
_iterations
_learning_rate

_momentums
_velocities*

trace_0* 

serving_default* 
]
_inbound_nodes
_outbound_nodes
_losses
	_loss_ids
_losses_override* 
�
cell
 _inbound_nodes
!_outbound_nodes
"_losses
#	_loss_ids
$_losses_override
%
state_size
&_build_shapes_dict*
�
'_kernel
(bias
)_inbound_nodes
*_outbound_nodes
+_losses
,	_loss_ids
-_losses_override
._build_shapes_dict*
u
/_inbound_nodes
0_outbound_nodes
1_losses
2	_loss_ids
3_losses_override
4_build_shapes_dict* 
Z
0
1
52
63
74
85
96
:7
;8
<9
=10
>11*
'
?0
@1
A2
'3
(4*
* 
UO
VARIABLE_VALUEVariable_160optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEVariable_153optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
�

?kernel
@recurrent_kernel
Abias
B_inbound_nodes
C_outbound_nodes
D_losses
E	_loss_ids
F_losses_override
G
state_size
H_build_shapes_dict*
* 
* 
* 
* 
* 
* 
* 
UO
VARIABLE_VALUEVariable_140_operations/2/_kernel/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEVariable_13-_operations/2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
VP
VARIABLE_VALUEVariable_121optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEVariable_111optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEVariable_101optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUE
Variable_91optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUE
Variable_81optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUE
Variable_71optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUE
Variable_61optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUE
Variable_51optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUE
Variable_42optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUE
Variable_32optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUE
Variable_2;optimizer/_trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUE
Variable_1;optimizer/_trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEVariable;optimizer/_trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameVariable_16Variable_15Variable_14Variable_13Variable_12Variable_11Variable_10
Variable_9
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1VariableConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU 2J 8� �J *'
f"R 
__inference__traced_save_37350
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable_16Variable_15Variable_14Variable_13Variable_12Variable_11Variable_10
Variable_9
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1Variable*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU 2J 8� �J **
f%R#
!__inference__traced_restore_37410�
�f
�
!__inference_serving_default_37142

inputsN
<functional_1_lstm_1_lstm_cell_1_cast_readvariableop_resource:PP
>functional_1_lstm_1_lstm_cell_1_cast_1_readvariableop_resource:PK
=functional_1_lstm_1_lstm_cell_1_add_1_readvariableop_resource:PC
1functional_1_dense_1_cast_readvariableop_resource:
B
4functional_1_dense_1_biasadd_readvariableop_resource:

identity��+functional_1/dense_1/BiasAdd/ReadVariableOp�(functional_1/dense_1/Cast/ReadVariableOp�3functional_1/lstm_1/lstm_cell_1/Cast/ReadVariableOp�5functional_1/lstm_1/lstm_cell_1/Cast_1/ReadVariableOp�4functional_1/lstm_1/lstm_cell_1/add_1/ReadVariableOp�functional_1/lstm_1/whilen
functional_1/lstm_1/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         n
functional_1/lstm_1/zerosConst*
_output_shapes

:*
dtype0*
valueB*    p
functional_1/lstm_1/zeros_1Const*
_output_shapes

:*
dtype0*
valueB*    |
'functional_1/lstm_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            ~
)functional_1/lstm_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           ~
)functional_1/lstm_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
!functional_1/lstm_1/strided_sliceStridedSliceinputs0functional_1/lstm_1/strided_slice/stack:output:02functional_1/lstm_1/strided_slice/stack_1:output:02functional_1/lstm_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask*
shrink_axis_maskw
"functional_1/lstm_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
functional_1/lstm_1/transpose	Transposeinputs+functional_1/lstm_1/transpose/perm:output:0*
T0*"
_output_shapes
:z
/functional_1/lstm_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������p
.functional_1/lstm_1/TensorArrayV2/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
!functional_1/lstm_1/TensorArrayV2TensorListReserve8functional_1/lstm_1/TensorArrayV2/element_shape:output:07functional_1/lstm_1/TensorArrayV2/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Ifunctional_1/lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
;functional_1/lstm_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!functional_1/lstm_1/transpose:y:0Rfunctional_1/lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���s
)functional_1/lstm_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+functional_1/lstm_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+functional_1/lstm_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#functional_1/lstm_1/strided_slice_1StridedSlice!functional_1/lstm_1/transpose:y:02functional_1/lstm_1/strided_slice_1/stack:output:04functional_1/lstm_1/strided_slice_1/stack_1:output:04functional_1/lstm_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask�
3functional_1/lstm_1/lstm_cell_1/Cast/ReadVariableOpReadVariableOp<functional_1_lstm_1_lstm_cell_1_cast_readvariableop_resource*
_output_shapes

:P*
dtype0�
&functional_1/lstm_1/lstm_cell_1/MatMulMatMul,functional_1/lstm_1/strided_slice_1:output:0;functional_1/lstm_1/lstm_cell_1/Cast/ReadVariableOp:value:0*
T0*
_output_shapes

:P�
5functional_1/lstm_1/lstm_cell_1/Cast_1/ReadVariableOpReadVariableOp>functional_1_lstm_1_lstm_cell_1_cast_1_readvariableop_resource*
_output_shapes

:P*
dtype0�
(functional_1/lstm_1/lstm_cell_1/MatMul_1MatMul"functional_1/lstm_1/zeros:output:0=functional_1/lstm_1/lstm_cell_1/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes

:P�
#functional_1/lstm_1/lstm_cell_1/addAddV20functional_1/lstm_1/lstm_cell_1/MatMul:product:02functional_1/lstm_1/lstm_cell_1/MatMul_1:product:0*
T0*
_output_shapes

:P�
4functional_1/lstm_1/lstm_cell_1/add_1/ReadVariableOpReadVariableOp=functional_1_lstm_1_lstm_cell_1_add_1_readvariableop_resource*
_output_shapes
:P*
dtype0�
%functional_1/lstm_1/lstm_cell_1/add_1AddV2'functional_1/lstm_1/lstm_cell_1/add:z:0<functional_1/lstm_1/lstm_cell_1/add_1/ReadVariableOp:value:0*
T0*
_output_shapes

:Pq
/functional_1/lstm_1/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
%functional_1/lstm_1/lstm_cell_1/splitSplit8functional_1/lstm_1/lstm_cell_1/split/split_dim:output:0)functional_1/lstm_1/lstm_cell_1/add_1:z:0*
T0*<
_output_shapes*
(::::*
	num_split�
'functional_1/lstm_1/lstm_cell_1/SigmoidSigmoid.functional_1/lstm_1/lstm_cell_1/split:output:0*
T0*
_output_shapes

:�
)functional_1/lstm_1/lstm_cell_1/Sigmoid_1Sigmoid.functional_1/lstm_1/lstm_cell_1/split:output:1*
T0*
_output_shapes

:�
#functional_1/lstm_1/lstm_cell_1/mulMul-functional_1/lstm_1/lstm_cell_1/Sigmoid_1:y:0$functional_1/lstm_1/zeros_1:output:0*
T0*
_output_shapes

:�
$functional_1/lstm_1/lstm_cell_1/ReluRelu.functional_1/lstm_1/lstm_cell_1/split:output:2*
T0*
_output_shapes

:�
%functional_1/lstm_1/lstm_cell_1/mul_1Mul+functional_1/lstm_1/lstm_cell_1/Sigmoid:y:02functional_1/lstm_1/lstm_cell_1/Relu:activations:0*
T0*
_output_shapes

:�
%functional_1/lstm_1/lstm_cell_1/add_2AddV2'functional_1/lstm_1/lstm_cell_1/mul:z:0)functional_1/lstm_1/lstm_cell_1/mul_1:z:0*
T0*
_output_shapes

:�
)functional_1/lstm_1/lstm_cell_1/Sigmoid_2Sigmoid.functional_1/lstm_1/lstm_cell_1/split:output:3*
T0*
_output_shapes

:�
&functional_1/lstm_1/lstm_cell_1/Relu_1Relu)functional_1/lstm_1/lstm_cell_1/add_2:z:0*
T0*
_output_shapes

:�
%functional_1/lstm_1/lstm_cell_1/mul_2Mul-functional_1/lstm_1/lstm_cell_1/Sigmoid_2:y:04functional_1/lstm_1/lstm_cell_1/Relu_1:activations:0*
T0*
_output_shapes

:�
1functional_1/lstm_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      r
0functional_1/lstm_1/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
#functional_1/lstm_1/TensorArrayV2_1TensorListReserve:functional_1/lstm_1/TensorArrayV2_1/element_shape:output:09functional_1/lstm_1/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���Z
functional_1/lstm_1/timeConst*
_output_shapes
: *
dtype0*
value	B : `
functional_1/lstm_1/Rank/ConstConst*
_output_shapes
: *
dtype0*
value	B :Z
functional_1/lstm_1/RankConst*
_output_shapes
: *
dtype0*
value	B : a
functional_1/lstm_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : a
functional_1/lstm_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
functional_1/lstm_1/rangeRange(functional_1/lstm_1/range/start:output:0!functional_1/lstm_1/Rank:output:0(functional_1/lstm_1/range/delta:output:0*
_output_shapes
: _
functional_1/lstm_1/Max/inputConst*
_output_shapes
: *
dtype0*
value	B :�
functional_1/lstm_1/MaxMax&functional_1/lstm_1/Max/input:output:0"functional_1/lstm_1/range:output:0*
T0*
_output_shapes
: h
&functional_1/lstm_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
functional_1/lstm_1/whileWhile/functional_1/lstm_1/while/loop_counter:output:0 functional_1/lstm_1/Max:output:0!functional_1/lstm_1/time:output:0,functional_1/lstm_1/TensorArrayV2_1:handle:0"functional_1/lstm_1/zeros:output:0$functional_1/lstm_1/zeros_1:output:0Kfunctional_1/lstm_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0<functional_1_lstm_1_lstm_cell_1_cast_readvariableop_resource>functional_1_lstm_1_lstm_cell_1_cast_1_readvariableop_resource=functional_1_lstm_1_lstm_cell_1_add_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*8
_output_shapes&
$: : : : ::: : : : *%
_read_only_resource_inputs
	*0
body(R&
$functional_1_lstm_1_while_body_37050*0
cond(R&
$functional_1_lstm_1_while_cond_37049*7
output_shapes&
$: : : : ::: : : : *
parallel_iterations �
Dfunctional_1/lstm_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
6functional_1/lstm_1/TensorArrayV2Stack/TensorListStackTensorListStack"functional_1/lstm_1/while:output:3Mfunctional_1/lstm_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:*
element_dtype0*
num_elements|
)functional_1/lstm_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������u
+functional_1/lstm_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+functional_1/lstm_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#functional_1/lstm_1/strided_slice_2StridedSlice?functional_1/lstm_1/TensorArrayV2Stack/TensorListStack:tensor:02functional_1/lstm_1/strided_slice_2/stack:output:04functional_1/lstm_1/strided_slice_2/stack_1:output:04functional_1/lstm_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_masky
$functional_1/lstm_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
functional_1/lstm_1/transpose_1	Transpose?functional_1/lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0-functional_1/lstm_1/transpose_1/perm:output:0*
T0*"
_output_shapes
:�
(functional_1/dense_1/Cast/ReadVariableOpReadVariableOp1functional_1_dense_1_cast_readvariableop_resource*
_output_shapes

:
*
dtype0�
functional_1/dense_1/MatMulMatMul,functional_1/lstm_1/strided_slice_2:output:00functional_1/dense_1/Cast/ReadVariableOp:value:0*
T0*
_output_shapes

:
�
+functional_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4functional_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
functional_1/dense_1/BiasAddBiasAdd%functional_1/dense_1/MatMul:product:03functional_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:
m
functional_1/reshape_1/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   
   y
$functional_1/reshape_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
functional_1/reshape_1/ReshapeReshape%functional_1/dense_1/BiasAdd:output:0-functional_1/reshape_1/Reshape/shape:output:0*
T0*"
_output_shapes
:q
IdentityIdentity'functional_1/reshape_1/Reshape:output:0^NoOp*
T0*"
_output_shapes
:�
NoOpNoOp,^functional_1/dense_1/BiasAdd/ReadVariableOp)^functional_1/dense_1/Cast/ReadVariableOp4^functional_1/lstm_1/lstm_cell_1/Cast/ReadVariableOp6^functional_1/lstm_1/lstm_cell_1/Cast_1/ReadVariableOp5^functional_1/lstm_1/lstm_cell_1/add_1/ReadVariableOp^functional_1/lstm_1/while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:: : : : : 2Z
+functional_1/dense_1/BiasAdd/ReadVariableOp+functional_1/dense_1/BiasAdd/ReadVariableOp2T
(functional_1/dense_1/Cast/ReadVariableOp(functional_1/dense_1/Cast/ReadVariableOp2j
3functional_1/lstm_1/lstm_cell_1/Cast/ReadVariableOp3functional_1/lstm_1/lstm_cell_1/Cast/ReadVariableOp2n
5functional_1/lstm_1/lstm_cell_1/Cast_1/ReadVariableOp5functional_1/lstm_1/lstm_cell_1/Cast_1/ReadVariableOp2l
4functional_1/lstm_1/lstm_cell_1/add_1/ReadVariableOp4functional_1/lstm_1/lstm_cell_1/add_1/ReadVariableOp26
functional_1/lstm_1/whilefunctional_1/lstm_1/while:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:J F
"
_output_shapes
:
 
_user_specified_nameinputs
��
�
__inference__traced_save_37350
file_prefix,
"read_disablecopyonread_variable_16:	 .
$read_1_disablecopyonread_variable_15: 6
$read_2_disablecopyonread_variable_14:
2
$read_3_disablecopyonread_variable_13:
6
$read_4_disablecopyonread_variable_12:P6
$read_5_disablecopyonread_variable_11:P6
$read_6_disablecopyonread_variable_10:P5
#read_7_disablecopyonread_variable_9:P1
#read_8_disablecopyonread_variable_8:P1
#read_9_disablecopyonread_variable_7:P6
$read_10_disablecopyonread_variable_6:
6
$read_11_disablecopyonread_variable_5:
2
$read_12_disablecopyonread_variable_4:
2
$read_13_disablecopyonread_variable_3:
6
$read_14_disablecopyonread_variable_2:P6
$read_15_disablecopyonread_variable_1:P0
"read_16_disablecopyonread_variable:P
savev2_const
identity_35��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: e
Read/DisableCopyOnReadDisableCopyOnRead"read_disablecopyonread_variable_16*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp"read_disablecopyonread_variable_16^Read/DisableCopyOnRead*
_output_shapes
: *
dtype0	R
IdentityIdentityRead/ReadVariableOp:value:0*
T0	*
_output_shapes
: Y

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0	*
_output_shapes
: i
Read_1/DisableCopyOnReadDisableCopyOnRead$read_1_disablecopyonread_variable_15*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp$read_1_disablecopyonread_variable_15^Read_1/DisableCopyOnRead*
_output_shapes
: *
dtype0V

Identity_2IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
: [

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
: i
Read_2/DisableCopyOnReadDisableCopyOnRead$read_2_disablecopyonread_variable_14*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp$read_2_disablecopyonread_variable_14^Read_2/DisableCopyOnRead*
_output_shapes

:
*
dtype0^

Identity_4IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes

:
c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:
i
Read_3/DisableCopyOnReadDisableCopyOnRead$read_3_disablecopyonread_variable_13*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp$read_3_disablecopyonread_variable_13^Read_3/DisableCopyOnRead*
_output_shapes
:
*
dtype0Z

Identity_6IdentityRead_3/ReadVariableOp:value:0*
T0*
_output_shapes
:
_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:
i
Read_4/DisableCopyOnReadDisableCopyOnRead$read_4_disablecopyonread_variable_12*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp$read_4_disablecopyonread_variable_12^Read_4/DisableCopyOnRead*
_output_shapes

:P*
dtype0^

Identity_8IdentityRead_4/ReadVariableOp:value:0*
T0*
_output_shapes

:Pc

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes

:Pi
Read_5/DisableCopyOnReadDisableCopyOnRead$read_5_disablecopyonread_variable_11*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp$read_5_disablecopyonread_variable_11^Read_5/DisableCopyOnRead*
_output_shapes

:P*
dtype0_
Identity_10IdentityRead_5/ReadVariableOp:value:0*
T0*
_output_shapes

:Pe
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes

:Pi
Read_6/DisableCopyOnReadDisableCopyOnRead$read_6_disablecopyonread_variable_10*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp$read_6_disablecopyonread_variable_10^Read_6/DisableCopyOnRead*
_output_shapes

:P*
dtype0_
Identity_12IdentityRead_6/ReadVariableOp:value:0*
T0*
_output_shapes

:Pe
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes

:Ph
Read_7/DisableCopyOnReadDisableCopyOnRead#read_7_disablecopyonread_variable_9*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp#read_7_disablecopyonread_variable_9^Read_7/DisableCopyOnRead*
_output_shapes

:P*
dtype0_
Identity_14IdentityRead_7/ReadVariableOp:value:0*
T0*
_output_shapes

:Pe
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes

:Ph
Read_8/DisableCopyOnReadDisableCopyOnRead#read_8_disablecopyonread_variable_8*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp#read_8_disablecopyonread_variable_8^Read_8/DisableCopyOnRead*
_output_shapes
:P*
dtype0[
Identity_16IdentityRead_8/ReadVariableOp:value:0*
T0*
_output_shapes
:Pa
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:Ph
Read_9/DisableCopyOnReadDisableCopyOnRead#read_9_disablecopyonread_variable_7*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp#read_9_disablecopyonread_variable_7^Read_9/DisableCopyOnRead*
_output_shapes
:P*
dtype0[
Identity_18IdentityRead_9/ReadVariableOp:value:0*
T0*
_output_shapes
:Pa
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:Pj
Read_10/DisableCopyOnReadDisableCopyOnRead$read_10_disablecopyonread_variable_6*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp$read_10_disablecopyonread_variable_6^Read_10/DisableCopyOnRead*
_output_shapes

:
*
dtype0`
Identity_20IdentityRead_10/ReadVariableOp:value:0*
T0*
_output_shapes

:
e
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes

:
j
Read_11/DisableCopyOnReadDisableCopyOnRead$read_11_disablecopyonread_variable_5*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp$read_11_disablecopyonread_variable_5^Read_11/DisableCopyOnRead*
_output_shapes

:
*
dtype0`
Identity_22IdentityRead_11/ReadVariableOp:value:0*
T0*
_output_shapes

:
e
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes

:
j
Read_12/DisableCopyOnReadDisableCopyOnRead$read_12_disablecopyonread_variable_4*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp$read_12_disablecopyonread_variable_4^Read_12/DisableCopyOnRead*
_output_shapes
:
*
dtype0\
Identity_24IdentityRead_12/ReadVariableOp:value:0*
T0*
_output_shapes
:
a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:
j
Read_13/DisableCopyOnReadDisableCopyOnRead$read_13_disablecopyonread_variable_3*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp$read_13_disablecopyonread_variable_3^Read_13/DisableCopyOnRead*
_output_shapes
:
*
dtype0\
Identity_26IdentityRead_13/ReadVariableOp:value:0*
T0*
_output_shapes
:
a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:
j
Read_14/DisableCopyOnReadDisableCopyOnRead$read_14_disablecopyonread_variable_2*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp$read_14_disablecopyonread_variable_2^Read_14/DisableCopyOnRead*
_output_shapes

:P*
dtype0`
Identity_28IdentityRead_14/ReadVariableOp:value:0*
T0*
_output_shapes

:Pe
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes

:Pj
Read_15/DisableCopyOnReadDisableCopyOnRead$read_15_disablecopyonread_variable_1*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp$read_15_disablecopyonread_variable_1^Read_15/DisableCopyOnRead*
_output_shapes

:P*
dtype0`
Identity_30IdentityRead_15/ReadVariableOp:value:0*
T0*
_output_shapes

:Pe
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes

:Ph
Read_16/DisableCopyOnReadDisableCopyOnRead"read_16_disablecopyonread_variable*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp"read_16_disablecopyonread_variable^Read_16/DisableCopyOnRead*
_output_shapes
:P*
dtype0\
Identity_32IdentityRead_16/ReadVariableOp:value:0*
T0*
_output_shapes
:Pa
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:PL

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0_operations/2/_kernel/.ATTRIBUTES/VARIABLE_VALUEB-_operations/2/bias/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*7
value.B,B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 * 
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_34Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_35IdentityIdentity_34:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_35Identity_35:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:($
"
_user_specified_name
Variable:*&
$
_user_specified_name
Variable_1:*&
$
_user_specified_name
Variable_2:*&
$
_user_specified_name
Variable_3:*&
$
_user_specified_name
Variable_4:*&
$
_user_specified_name
Variable_5:*&
$
_user_specified_name
Variable_6:*
&
$
_user_specified_name
Variable_7:*	&
$
_user_specified_name
Variable_8:*&
$
_user_specified_name
Variable_9:+'
%
_user_specified_nameVariable_10:+'
%
_user_specified_nameVariable_11:+'
%
_user_specified_nameVariable_12:+'
%
_user_specified_nameVariable_13:+'
%
_user_specified_nameVariable_14:+'
%
_user_specified_nameVariable_15:+'
%
_user_specified_nameVariable_16:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�M
�
$functional_1_lstm_1_while_body_37050D
@functional_1_lstm_1_while_functional_1_lstm_1_while_loop_counter5
1functional_1_lstm_1_while_functional_1_lstm_1_max)
%functional_1_lstm_1_while_placeholder+
'functional_1_lstm_1_while_placeholder_1+
'functional_1_lstm_1_while_placeholder_2+
'functional_1_lstm_1_while_placeholder_3
{functional_1_lstm_1_while_tensorarrayv2read_tensorlistgetitem_functional_1_lstm_1_tensorarrayunstack_tensorlistfromtensor_0V
Dfunctional_1_lstm_1_while_lstm_cell_1_cast_readvariableop_resource_0:PX
Ffunctional_1_lstm_1_while_lstm_cell_1_cast_1_readvariableop_resource_0:PS
Efunctional_1_lstm_1_while_lstm_cell_1_add_1_readvariableop_resource_0:P&
"functional_1_lstm_1_while_identity(
$functional_1_lstm_1_while_identity_1(
$functional_1_lstm_1_while_identity_2(
$functional_1_lstm_1_while_identity_3(
$functional_1_lstm_1_while_identity_4(
$functional_1_lstm_1_while_identity_5}
yfunctional_1_lstm_1_while_tensorarrayv2read_tensorlistgetitem_functional_1_lstm_1_tensorarrayunstack_tensorlistfromtensorT
Bfunctional_1_lstm_1_while_lstm_cell_1_cast_readvariableop_resource:PV
Dfunctional_1_lstm_1_while_lstm_cell_1_cast_1_readvariableop_resource:PQ
Cfunctional_1_lstm_1_while_lstm_cell_1_add_1_readvariableop_resource:P��9functional_1/lstm_1/while/lstm_cell_1/Cast/ReadVariableOp�;functional_1/lstm_1/while/lstm_cell_1/Cast_1/ReadVariableOp�:functional_1/lstm_1/while/lstm_cell_1/add_1/ReadVariableOp�
Kfunctional_1/lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
=functional_1/lstm_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{functional_1_lstm_1_while_tensorarrayv2read_tensorlistgetitem_functional_1_lstm_1_tensorarrayunstack_tensorlistfromtensor_0%functional_1_lstm_1_while_placeholderTfunctional_1/lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype0�
9functional_1/lstm_1/while/lstm_cell_1/Cast/ReadVariableOpReadVariableOpDfunctional_1_lstm_1_while_lstm_cell_1_cast_readvariableop_resource_0*
_output_shapes

:P*
dtype0�
,functional_1/lstm_1/while/lstm_cell_1/MatMulMatMulDfunctional_1/lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0Afunctional_1/lstm_1/while/lstm_cell_1/Cast/ReadVariableOp:value:0*
T0*
_output_shapes

:P�
;functional_1/lstm_1/while/lstm_cell_1/Cast_1/ReadVariableOpReadVariableOpFfunctional_1_lstm_1_while_lstm_cell_1_cast_1_readvariableop_resource_0*
_output_shapes

:P*
dtype0�
.functional_1/lstm_1/while/lstm_cell_1/MatMul_1MatMul'functional_1_lstm_1_while_placeholder_2Cfunctional_1/lstm_1/while/lstm_cell_1/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes

:P�
)functional_1/lstm_1/while/lstm_cell_1/addAddV26functional_1/lstm_1/while/lstm_cell_1/MatMul:product:08functional_1/lstm_1/while/lstm_cell_1/MatMul_1:product:0*
T0*
_output_shapes

:P�
:functional_1/lstm_1/while/lstm_cell_1/add_1/ReadVariableOpReadVariableOpEfunctional_1_lstm_1_while_lstm_cell_1_add_1_readvariableop_resource_0*
_output_shapes
:P*
dtype0�
+functional_1/lstm_1/while/lstm_cell_1/add_1AddV2-functional_1/lstm_1/while/lstm_cell_1/add:z:0Bfunctional_1/lstm_1/while/lstm_cell_1/add_1/ReadVariableOp:value:0*
T0*
_output_shapes

:Pw
5functional_1/lstm_1/while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
+functional_1/lstm_1/while/lstm_cell_1/splitSplit>functional_1/lstm_1/while/lstm_cell_1/split/split_dim:output:0/functional_1/lstm_1/while/lstm_cell_1/add_1:z:0*
T0*<
_output_shapes*
(::::*
	num_split�
-functional_1/lstm_1/while/lstm_cell_1/SigmoidSigmoid4functional_1/lstm_1/while/lstm_cell_1/split:output:0*
T0*
_output_shapes

:�
/functional_1/lstm_1/while/lstm_cell_1/Sigmoid_1Sigmoid4functional_1/lstm_1/while/lstm_cell_1/split:output:1*
T0*
_output_shapes

:�
)functional_1/lstm_1/while/lstm_cell_1/mulMul3functional_1/lstm_1/while/lstm_cell_1/Sigmoid_1:y:0'functional_1_lstm_1_while_placeholder_3*
T0*
_output_shapes

:�
*functional_1/lstm_1/while/lstm_cell_1/ReluRelu4functional_1/lstm_1/while/lstm_cell_1/split:output:2*
T0*
_output_shapes

:�
+functional_1/lstm_1/while/lstm_cell_1/mul_1Mul1functional_1/lstm_1/while/lstm_cell_1/Sigmoid:y:08functional_1/lstm_1/while/lstm_cell_1/Relu:activations:0*
T0*
_output_shapes

:�
+functional_1/lstm_1/while/lstm_cell_1/add_2AddV2-functional_1/lstm_1/while/lstm_cell_1/mul:z:0/functional_1/lstm_1/while/lstm_cell_1/mul_1:z:0*
T0*
_output_shapes

:�
/functional_1/lstm_1/while/lstm_cell_1/Sigmoid_2Sigmoid4functional_1/lstm_1/while/lstm_cell_1/split:output:3*
T0*
_output_shapes

:�
,functional_1/lstm_1/while/lstm_cell_1/Relu_1Relu/functional_1/lstm_1/while/lstm_cell_1/add_2:z:0*
T0*
_output_shapes

:�
+functional_1/lstm_1/while/lstm_cell_1/mul_2Mul3functional_1/lstm_1/while/lstm_cell_1/Sigmoid_2:y:0:functional_1/lstm_1/while/lstm_cell_1/Relu_1:activations:0*
T0*
_output_shapes

:�
Dfunctional_1/lstm_1/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
>functional_1/lstm_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'functional_1_lstm_1_while_placeholder_1Mfunctional_1/lstm_1/while/TensorArrayV2Write/TensorListSetItem/index:output:0/functional_1/lstm_1/while/lstm_cell_1/mul_2:z:0*
_output_shapes
: *
element_dtype0:���a
functional_1/lstm_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
functional_1/lstm_1/while/addAddV2%functional_1_lstm_1_while_placeholder(functional_1/lstm_1/while/add/y:output:0*
T0*
_output_shapes
: c
!functional_1/lstm_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
functional_1/lstm_1/while/add_1AddV2@functional_1_lstm_1_while_functional_1_lstm_1_while_loop_counter*functional_1/lstm_1/while/add_1/y:output:0*
T0*
_output_shapes
: �
"functional_1/lstm_1/while/IdentityIdentity#functional_1/lstm_1/while/add_1:z:0^functional_1/lstm_1/while/NoOp*
T0*
_output_shapes
: �
$functional_1/lstm_1/while/Identity_1Identity1functional_1_lstm_1_while_functional_1_lstm_1_max^functional_1/lstm_1/while/NoOp*
T0*
_output_shapes
: �
$functional_1/lstm_1/while/Identity_2Identity!functional_1/lstm_1/while/add:z:0^functional_1/lstm_1/while/NoOp*
T0*
_output_shapes
: �
$functional_1/lstm_1/while/Identity_3IdentityNfunctional_1/lstm_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^functional_1/lstm_1/while/NoOp*
T0*
_output_shapes
: �
$functional_1/lstm_1/while/Identity_4Identity/functional_1/lstm_1/while/lstm_cell_1/mul_2:z:0^functional_1/lstm_1/while/NoOp*
T0*
_output_shapes

:�
$functional_1/lstm_1/while/Identity_5Identity/functional_1/lstm_1/while/lstm_cell_1/add_2:z:0^functional_1/lstm_1/while/NoOp*
T0*
_output_shapes

:�
functional_1/lstm_1/while/NoOpNoOp:^functional_1/lstm_1/while/lstm_cell_1/Cast/ReadVariableOp<^functional_1/lstm_1/while/lstm_cell_1/Cast_1/ReadVariableOp;^functional_1/lstm_1/while/lstm_cell_1/add_1/ReadVariableOp*
_output_shapes
 "U
$functional_1_lstm_1_while_identity_1-functional_1/lstm_1/while/Identity_1:output:0"U
$functional_1_lstm_1_while_identity_2-functional_1/lstm_1/while/Identity_2:output:0"U
$functional_1_lstm_1_while_identity_3-functional_1/lstm_1/while/Identity_3:output:0"U
$functional_1_lstm_1_while_identity_4-functional_1/lstm_1/while/Identity_4:output:0"U
$functional_1_lstm_1_while_identity_5-functional_1/lstm_1/while/Identity_5:output:0"Q
"functional_1_lstm_1_while_identity+functional_1/lstm_1/while/Identity:output:0"�
Cfunctional_1_lstm_1_while_lstm_cell_1_add_1_readvariableop_resourceEfunctional_1_lstm_1_while_lstm_cell_1_add_1_readvariableop_resource_0"�
Dfunctional_1_lstm_1_while_lstm_cell_1_cast_1_readvariableop_resourceFfunctional_1_lstm_1_while_lstm_cell_1_cast_1_readvariableop_resource_0"�
Bfunctional_1_lstm_1_while_lstm_cell_1_cast_readvariableop_resourceDfunctional_1_lstm_1_while_lstm_cell_1_cast_readvariableop_resource_0"�
yfunctional_1_lstm_1_while_tensorarrayv2read_tensorlistgetitem_functional_1_lstm_1_tensorarrayunstack_tensorlistfromtensor{functional_1_lstm_1_while_tensorarrayv2read_tensorlistgetitem_functional_1_lstm_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$: : : : ::: : : : 2v
9functional_1/lstm_1/while/lstm_cell_1/Cast/ReadVariableOp9functional_1/lstm_1/while/lstm_cell_1/Cast/ReadVariableOp2z
;functional_1/lstm_1/while/lstm_cell_1/Cast_1/ReadVariableOp;functional_1/lstm_1/while/lstm_cell_1/Cast_1/ReadVariableOp2x
:functional_1/lstm_1/while/lstm_cell_1/add_1/ReadVariableOp:functional_1/lstm_1/while/lstm_cell_1/add_1/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:so

_output_shapes
: 
U
_user_specified_name=;functional_1/lstm_1/TensorArrayUnstack/TensorListFromTensor:$ 

_output_shapes

::$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :OK

_output_shapes
: 
1
_user_specified_namefunctional_1/lstm_1/Max:^ Z

_output_shapes
: 
@
_user_specified_name(&functional_1/lstm_1/while/loop_counter
�	
�
3__inference_signature_wrapper_serving_default_37158

inputs
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:

	unknown_3:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*'
_read_only_resource_inputs	
*2
config_proto" 

CPU

GPU 2J 8� �J **
f%R#
!__inference_serving_default_37142j
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*"
_output_shapes
:<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name37154:%!

_user_specified_name37152:%!

_user_specified_name37150:%!

_user_specified_name37148:%!

_user_specified_name37146:J F
"
_output_shapes
:
 
_user_specified_nameinputs
�O
�	
!__inference__traced_restore_37410
file_prefix&
assignvariableop_variable_16:	 (
assignvariableop_1_variable_15: 0
assignvariableop_2_variable_14:
,
assignvariableop_3_variable_13:
0
assignvariableop_4_variable_12:P0
assignvariableop_5_variable_11:P0
assignvariableop_6_variable_10:P/
assignvariableop_7_variable_9:P+
assignvariableop_8_variable_8:P+
assignvariableop_9_variable_7:P0
assignvariableop_10_variable_6:
0
assignvariableop_11_variable_5:
,
assignvariableop_12_variable_4:
,
assignvariableop_13_variable_3:
0
assignvariableop_14_variable_2:P0
assignvariableop_15_variable_1:P*
assignvariableop_16_variable:P
identity_18��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0_operations/2/_kernel/.ATTRIBUTES/VARIABLE_VALUEB-_operations/2/bias/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*7
value.B,B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*\
_output_shapesJ
H::::::::::::::::::* 
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_variable_16Identity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_15Identity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_variable_14Identity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_variable_13Identity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_variable_12Identity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_variable_11Identity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_variable_10Identity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_variable_9Identity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_variable_8Identity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_variable_7Identity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_variable_6Identity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_variable_5Identity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_variable_4Identity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_variable_3Identity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_variable_2Identity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_variable_1Identity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_variableIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_17Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_18IdentityIdentity_17:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_18Identity_18:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$: : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:($
"
_user_specified_name
Variable:*&
$
_user_specified_name
Variable_1:*&
$
_user_specified_name
Variable_2:*&
$
_user_specified_name
Variable_3:*&
$
_user_specified_name
Variable_4:*&
$
_user_specified_name
Variable_5:*&
$
_user_specified_name
Variable_6:*
&
$
_user_specified_name
Variable_7:*	&
$
_user_specified_name
Variable_8:*&
$
_user_specified_name
Variable_9:+'
%
_user_specified_nameVariable_10:+'
%
_user_specified_nameVariable_11:+'
%
_user_specified_nameVariable_12:+'
%
_user_specified_nameVariable_13:+'
%
_user_specified_nameVariable_14:+'
%
_user_specified_nameVariable_15:+'
%
_user_specified_nameVariable_16:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
$functional_1_lstm_1_while_cond_37049D
@functional_1_lstm_1_while_functional_1_lstm_1_while_loop_counter5
1functional_1_lstm_1_while_functional_1_lstm_1_max)
%functional_1_lstm_1_while_placeholder+
'functional_1_lstm_1_while_placeholder_1+
'functional_1_lstm_1_while_placeholder_2+
'functional_1_lstm_1_while_placeholder_3[
Wfunctional_1_lstm_1_while_functional_1_lstm_1_while_cond_37049___redundant_placeholder0[
Wfunctional_1_lstm_1_while_functional_1_lstm_1_while_cond_37049___redundant_placeholder1[
Wfunctional_1_lstm_1_while_functional_1_lstm_1_while_cond_37049___redundant_placeholder2[
Wfunctional_1_lstm_1_while_functional_1_lstm_1_while_cond_37049___redundant_placeholder3&
"functional_1_lstm_1_while_identity
b
 functional_1/lstm_1/while/Less/yConst*
_output_shapes
: *
dtype0*
value	B :�
functional_1/lstm_1/while/LessLess%functional_1_lstm_1_while_placeholder)functional_1/lstm_1/while/Less/y:output:0*
T0*
_output_shapes
: �
 functional_1/lstm_1/while/Less_1Less@functional_1_lstm_1_while_functional_1_lstm_1_while_loop_counter1functional_1_lstm_1_while_functional_1_lstm_1_max*
T0*
_output_shapes
: �
$functional_1/lstm_1/while/LogicalAnd
LogicalAnd$functional_1/lstm_1/while/Less_1:z:0"functional_1/lstm_1/while/Less:z:0*
_output_shapes
: y
"functional_1/lstm_1/while/IdentityIdentity(functional_1/lstm_1/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "Q
"functional_1_lstm_1_while_identity+functional_1/lstm_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,: : : : :::::::

_output_shapes
::$ 

_output_shapes

::$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :OK

_output_shapes
: 
1
_user_specified_namefunctional_1/lstm_1/Max:^ Z

_output_shapes
: 
@
_user_specified_name(&functional_1/lstm_1/while/loop_counter"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
4
inputs*
serving_default_inputs:07
output_0+
StatefulPartitionedCall:0tensorflow/serving/predict:�$
�
_tracked
_inbound_nodes
_outbound_nodes
_losses
_losses_override
_operations
_layers
_build_shapes_dict
	output_names

	optimizer
_default_save_signature

signatures"
_generic_user_object
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
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
�

_variables
_trainable_variables
 _trainable_variables_indices
_iterations
_learning_rate

_momentums
_velocities"
_generic_user_object
�
trace_02�
!__inference_serving_default_37142�
���
FullArgSpec
args�

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
annotations� *�
�ztrace_0
,
serving_default"
signature_map
y
_inbound_nodes
_outbound_nodes
_losses
	_loss_ids
_losses_override"
_generic_user_object
�
cell
 _inbound_nodes
!_outbound_nodes
"_losses
#	_loss_ids
$_losses_override
%
state_size
&_build_shapes_dict"
_generic_user_object
�
'_kernel
(bias
)_inbound_nodes
*_outbound_nodes
+_losses
,	_loss_ids
-_losses_override
._build_shapes_dict"
_generic_user_object
�
/_inbound_nodes
0_outbound_nodes
1_losses
2	_loss_ids
3_losses_override
4_build_shapes_dict"
_generic_user_object
v
0
1
52
63
74
85
96
:7
;8
<9
=10
>11"
trackable_list_wrapper
C
?0
@1
A2
'3
(4"
trackable_list_wrapper
 "
trackable_dict_wrapper
:	 (2adam/iteration
: (2adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�B�
!__inference_serving_default_37142inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
3__inference_signature_wrapper_serving_default_37158inputs"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs�

jinputs
kwonlydefaults
 
annotations� *
 
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
�

?kernel
@recurrent_kernel
Abias
B_inbound_nodes
C_outbound_nodes
D_losses
E	_loss_ids
F_losses_override
G
state_size
H_build_shapes_dict"
_generic_user_object
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
 "
trackable_dict_wrapper
 :
(2dense/kernel
:
(2
dense/bias
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
trackable_dict_wrapper
5:3P(2#adam/lstm_lstm_cell_kernel_momentum
5:3P(2#adam/lstm_lstm_cell_kernel_velocity
?:=P(2-adam/lstm_lstm_cell_recurrent_kernel_momentum
?:=P(2-adam/lstm_lstm_cell_recurrent_kernel_velocity
/:-P(2!adam/lstm_lstm_cell_bias_momentum
/:-P(2!adam/lstm_lstm_cell_bias_velocity
,:*
(2adam/dense_kernel_momentum
,:*
(2adam/dense_kernel_velocity
&:$
(2adam/dense_bias_momentum
&:$
(2adam/dense_bias_velocity
):'P(2lstm/lstm_cell/kernel
3:1P(2lstm/lstm_cell/recurrent_kernel
#:!P(2lstm/lstm_cell/bias
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
 "
trackable_dict_wrapperv
!__inference_serving_default_37142Q?@A'(*�'
 �
�
inputs
� "�
unknown�
3__inference_signature_wrapper_serving_default_37158m?@A'(4�1
� 
*�'
%
inputs�
inputs".�+
)
output_0�
output_0