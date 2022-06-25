# Owner(s): ["oncall: fx"]

import unittest
from torch.fx import GraphModule, symbolic_trace
from torch.fx.experimental.migrate_gradual_types.constraint import BinConstraintT, DVar, TVar, ApplyBroadcasting, T
from torch.fx.experimental.migrate_gradual_types.constraint_generator import ConstraintGenerator
from torch.fx.experimental.migrate_gradual_types.constraint_transformation import transform_constraint, \
    transform_apply_broadcasting
from torch.fx.experimental.migrate_gradual_types.operation import op_precision, op_matching, op_consistency
from torch.fx.experimental.migrate_gradual_types.transform_to_z3 import transform_all_constraints
from torch.fx.experimental.migrate_gradual_types.z3_types import tensor_type, D
from torch.fx.experimental.rewriter import RewritingTracer
from torch.fx.tensor_type import Dyn, TensorType
import torch


try:
    import z3  # type: ignore[import]
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False


try:
    from torchvision import models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")




class ComposeOperationsGradualTypes(unittest.TestCase):

    def test_add_reshape_1(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super(BasicBlock, self).__init__()

            def forward(self, x: Dyn, y: Dyn):
                return torch.add(torch.reshape(x, (1, 2)), torch.reshape(y, (2, 2)))

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        assert s.check() == z3.sat

    def test_add_reshape_2(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super(BasicBlock, self).__init__()

            def forward(self, x: Dyn, y: Dyn):
                return torch.add(torch.reshape(x, (-1, 2)), torch.reshape(y, (2, 2, 2)))

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        assert s.check() == z3.sat

    def test_conv_reshape_add_0(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self, in_planes, out_planes, kernel_size, stride, padding, groups, dilation):
                super(BasicBlock, self).__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                                             kernel_size=kernel_size, stride=stride,
                                             padding=padding, groups=groups, bias=False, dilation=dilation)

            def forward(self, x: Dyn, y: Dyn):
                return torch.add(self.conv1(torch.reshape(x, (1, 2, 10, 20))), y)

        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        new_transformed_c = transform_all_constraints(traced)
        solver = z3.Solver()
        solver.add(new_transformed_c)
        assert solver.check() == z3.sat


    def test_conv_reshape_add_0_2(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self, in_planes, out_planes, kernel_size, stride, padding, groups, dilation):
                super(BasicBlock, self).__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                                             kernel_size=kernel_size, stride=stride,
                                             padding=padding, groups=groups, bias=False, dilation=dilation)

            def forward(self, x: Dyn, y: TensorType([4, 1])):
                return torch.add(self.conv1(torch.reshape(x, (1, 2, 10, 20))), y)

        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)

        #        4,1
        # 1, 2, 4, 8
        res = B.forward(torch.rand(20, 20), torch.rand(1, 2, 4, 8)).size()
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        new_transformed_c = transform_all_constraints(traced)
        solver = z3.Solver()
        solver.add(new_transformed_c)
        assert solver.check() == z3.sat

        conv_result = z3.Const(4, tensor_type)
        add_result = z3.Const(9, tensor_type)
        input_2 = z3.Const(2, tensor_type)

        s1, s2, s3, s4 = z3.Ints('x1 x2 x3 x4')
        s11, s22, s33, s44 = z3.Ints('x11 x22 x33 x44')
        d1, d2, d3, d4 = D(s11, s1), D(s22, s2), D(s33, s3), D(s44, s4),


        solver.add(conv_result == tensor_type.tensor4(d1, d2, d3, d4))
        solver.check()
        assert solver.model()[s1].as_long() == res[0]
        assert solver.model()[s2].as_long() == res[1]
        assert solver.model()[s3].as_long() == res[2]
        assert solver.model()[s4].as_long() == res[3]

        solver.add(input_2 == tensor_type.tensor2(D(1, 4), D(1, 1)))
        assert solver.check() == z3.sat
        solver.add(add_result == tensor_type.tensor4(d1, d2, d3, d4))
        assert solver.check() == z3.sat

        # first dimension could be anything because we have broadcasting
        assert solver.model()[s1] == res[0]
        assert solver.model()[s2] == res[1]
        assert solver.model()[s3] == res[2]
        assert solver.model()[s4] == res[3]

    def test_conv_reshape_add_0_3(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self, in_planes, out_planes, kernel_size, stride, padding, groups, dilation):
                super(BasicBlock, self).__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                                             kernel_size=kernel_size, stride=stride,
                                             padding=padding, groups=groups, bias=False, dilation=dilation)

            def forward(self, x: Dyn, y: TensorType([11, 1])):
                return torch.add(self.conv1(torch.reshape(x, (1, 2, 10, 20))), y)

        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        new_transformed_c = transform_all_constraints(traced)
        solver = z3.Solver()
        solver.add(new_transformed_c)
        assert solver.check() == z3.unsat


    def test_conv_reshape_add_1(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self, in_planes, out_planes, kernel_size, stride, padding, groups, dilation):
                super(BasicBlock, self).__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                                             kernel_size=kernel_size, stride=stride,
                                             padding=padding, groups=groups, bias=False, dilation=dilation)

            def forward(self, x: Dyn, y: TensorType([1, 2, 10, 20])):
                return torch.add(self.conv1(torch.reshape(x, (1, 2, 10, 20))), y)

        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        new_transformed_c = transform_all_constraints(traced)
        solver = z3.Solver()
        solver.add(new_transformed_c)
        assert solver.check() == z3.unsat


class GradualTypes(unittest.TestCase):
    def test_conv_reshape_unsat(self):

        class BasicBlock(torch.nn.Module):
            def __init__(self, in_planes, out_planes, kernel_size, stride, padding, groups, dilation):
                super(BasicBlock, self).__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                                             kernel_size=kernel_size, stride=stride,
                                             padding=padding, groups=groups, bias=False, dilation=dilation)

            def forward(self, x: Dyn):
                return self.conv1(torch.reshape(x, (1, 2, 10)))

        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        new_transformed_c = transform_all_constraints(traced)
        solver = z3.Solver()
        solver.add(new_transformed_c)
        assert solver.check() == z3.unsat

    def test_conv_reshape0(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self, in_planes, out_planes, kernel_size, stride, padding, groups, dilation):
                super(BasicBlock, self).__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                                             kernel_size=kernel_size, stride=stride,
                                             padding=padding, groups=groups, bias=False, dilation=dilation)

            def forward(self, x: Dyn):
                return self.conv1(torch.reshape(x, (1, 2, 10, 20)))

        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)
        res = B.forward(torch.rand(20, 20)).size()
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        new_transformed_c = transform_all_constraints(traced)

        solver = z3.Solver()
        solver.add(new_transformed_c)
        assert solver.check() == z3.sat
        conv_result = z3.Const(3, tensor_type)

        s1, s2, s3, s4 = z3.Ints('x1 x2 x3 x4')
        s11, s22, s33, s44 = z3.Ints('x11 x22 x33 x44')
        d1, d2, d3, d4 = D(s11, s1), D(s22, s2), D(s33, s3), D(s44, s4),

        solver.add(conv_result == tensor_type.tensor4(d1, d2, d3, d4))
        solver.check()
        # print(solver.model())
        # print(type(solver.model()[s1]))
        assert solver.model()[s1].as_long() == res[0]
        assert solver.model()[s2].as_long() == res[1]
        assert solver.model()[s3].as_long() == res[2]
        assert solver.model()[s4].as_long() == res[3]

        s1, s2, s3, s4 = z3.Ints('y1 y2 y3 y4')
        s11, s22, s33, s44 = z3.Ints('y11 y22 y33 y44')
        d1, d2, d3, d4 = D(s11, s1), D(s22, s2), D(s33, s3), D(s44, s4),

        input = z3.Const(1, tensor_type)
        solver.add(input == tensor_type.tensor4(d1, d2, d3, d4))

        # assert solver.check() == sat
        # solver.add(s11 == 1)
        # solver.add(s22 == 1)
        # solver.add(s33 == 1)
        # solver.add(s44 == 1)
        #
        # print(solver.check())
        # print(solver.model())


    def test_conv_reshape1(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self, in_planes, out_planes, kernel_size, stride, padding, groups, dilation):
                super(BasicBlock, self).__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                                             kernel_size=kernel_size, stride=stride,
                                             padding=padding, groups=groups, bias=False, dilation=dilation)

            def forward(self, x: TensorType([20, 20])):
                return self.conv1(torch.reshape(x, (1, -1, 10, 20)))

        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)
        res = B.forward(torch.rand(20, 20)).size()
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        new_transformed_c = transform_all_constraints(traced)

        solver = z3.Solver()
        solver.add(new_transformed_c)
        assert solver.check() == z3.sat
        conv_result = z3.Const(3, tensor_type)

        s1, s2, s3, s4 = z3.Ints('x1 x2 x3 x4')
        s11, s22, s33, s44 = z3.Ints('x11 x22 x33 x44')
        d1, d2, d3, d4 = D(s11, s1), D(s22, s2), D(s33, s3), D(s44, s4),

        solver.add(conv_result == tensor_type.tensor4(d1, d2, d3, d4))
        solver.check()
        # print(solver.model())
        assert solver.model()[s1].as_long() == res[0]
        assert solver.model()[s2].as_long() == res[1]
        assert solver.model()[s3].as_long() == res[2]
        assert solver.model()[s4].as_long() == res[3]


class TestSingleOperation(unittest.TestCase):
    def test_conv_dyn(self):

        s1, s2, s3, s4 = z3.Ints('s1 s2 s3 s4')
        e1, e2, e3, e4 = z3.Ints('e1 e2 e3 e4')
        s11, s22, s33, s44 = z3.Ints('s11 s22 s33 s44')
        e11, e22, e33, e44 = z3.Ints('e11 e22 e33 e44')
        d1, d2, d3, d4 = D(s11, s1), D(s22, s2), D(s33, s3), D(s44, s4),
        b1, b2, b3, b4 = D(e11, e1), D(e22, e2), D(e33, e3), D(e44, e4)

        class BasicBlock(torch.nn.Module):
            def __init__(self, in_planes, out_planes, kernel_size, stride, padding, groups, dilation):
                super(BasicBlock, self).__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                                             kernel_size=kernel_size, stride=stride,
                                             padding=padding, groups=groups, bias=False, dilation=dilation)

            def forward(self, x: Dyn):
                return self.conv1(x)


        BasicBlock(2, 2, 2, 2, 2, 2, 2).forward(torch.rand(4, 2, 3, 4))

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock(2, 2, 2, 2, 2, 2, 2))
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced)

        solver3 = z3.Solver()
        solver3.add(transformed)
        assert solver3.check() == z3.sat

        x = z3.Const(1, tensor_type)
        y = z3.Const(2, tensor_type)

        solver3.add(x == tensor_type.tensor4(d1, d2, d3, d4),
                    y == tensor_type.tensor4(b1, b2, b3, b4))

        assert solver3.check() == z3.sat
        assert solver3.model()[s1].as_long() == solver3.model()[e1].as_long()
        assert solver3.model()[s11].as_long() == solver3.model()[e11].as_long()

        solver3.add(s2 != 2)
        assert solver3.check() == z3.sat
        assert solver3.model()[s22].as_long() == 0

        solver3.add(s22 != 0)
        assert solver3.check() == z3.unsat

        solver2 = z3.Solver()
        solver2.add(transformed)
        assert solver2.check() == z3.sat
        solver2.add(x == tensor_type.tensor3(d1, d2, d3))
        assert solver2.check() == z3.unsat


    def test_add(self):
        s1, s2, s3, s4 = z3.Ints('s1 s2 s3 s4')
        s11, s22, s33, s44 = z3.Ints('s11 s22 s33 s44')
        d1, d2, d3, d4 = D(s11, s1), D(s22, s2), D(s33, s3), D(s44, s4),

        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super(BasicBlock, self).__init__()

            def forward(self, x: Dyn, y: Dyn):
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        assert s.check() == z3.sat

        # make the tensor be of size 1
        x = z3.Const(1, tensor_type)
        s.add(x == tensor_type.tensor1(D(1, s11)))
        assert s.check() == z3.sat

        y = z3.Const(2, tensor_type)
        s.add(y == tensor_type.tensor1(D(1, s22)))
        assert s.check() == z3.sat

        s.add(s11 == 1)  # tensor[1]
        s.add(s22 == 2)  # tensor[2]
        assert s.check() == z3.sat

        class BasicBlock2(torch.nn.Module):
            def __init__(self):
                super(BasicBlock2, self).__init__()

            def forward(self, x: TensorType((Dyn,)), y: Dyn):
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock2())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced)
        s = z3.Solver()
        s.add(transformed)
        assert s.check() == z3.sat
        # make the tensor be of size 1
        x = z3.Const(1, tensor_type)
        s.add(x == tensor_type.tensor1(D(1, s11)))
        assert s.check() == z3.sat
        y = z3.Const(2, tensor_type)
        s.add(y == tensor_type.tensor1(D(1, s22)))
        assert s.check() == z3.sat
        s.add(s11 == 4)  # tensor[4]
        s.add(s22 == 5)  # tensor[5]
        assert s.check() == z3.unsat

        class BasicBlock3(torch.nn.Module):
            def __init__(self):
                super(BasicBlock3, self).__init__()

            def forward(self, x: TensorType((Dyn,)), y: Dyn):
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock3())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced)
        s = z3.Solver()
        s.add(transformed)
        x = z3.Const(1, tensor_type)
        s.add(x == tensor_type.tensor2(d1, d2))
        assert s.check() == z3.unsat

    def test_add_padding(self):
        s1, s2, s3, s4 = z3.Ints('s1 s2 s3 s4')

        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super(BasicBlock, self).__init__()

            def forward(self, x: TensorType((Dyn,)), y: TensorType((Dyn, Dyn))):
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        assert s.check() == z3.sat

        x = z3.Const(1, tensor_type)
        s.add(x == tensor_type.tensor1(D(1, s1)))

        assert s.check() == z3.sat

        # print(s.model())

    def test_add_padding_2(self):
        s1, s2, s3, s4 = z3.Ints('s1 s2 s3 s4')

        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super(BasicBlock, self).__init__()

            def forward(self, x: TensorType([Dyn, Dyn]), y: TensorType([Dyn])):
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        assert s.check() == z3.sat
        # print(s.model())

        x = z3.Const(1, tensor_type)
        s.add(x == tensor_type.tensor2(D(1, s1), D(1, s2)))
        assert s.check() == z3.sat

        y = z3.Const(2, tensor_type)
        s.add(y == tensor_type.tensor1(D(0, s3)))
        assert s.check() == z3.sat

        add_result = z3.Const(3, tensor_type)
        broadcast_res1, broadcast_res2 = z3.Const(4, tensor_type), z3.Const(5, tensor_type)

        # print(s.model())

        assert s.model()[broadcast_res1].decl() == tensor_type.tensor2
        assert s.model()[broadcast_res2].decl() == tensor_type.tensor2
        assert s.model()[add_result].decl() == tensor_type.tensor2
        assert s.model()[y].decl() == tensor_type.tensor1

        # print(s.model())

        # prevent broadcasting for that dimension
        s.add(s2 > 1)

        assert s.check()

        # the second dimension of the result is a number, not Dyn.
        # however if the first input dimension had been 1, we would
        # have had dyn in the result, as seen in the next test case
        assert s.model()[add_result].arg(1).arg(0).as_long() != 0

    def test_add_padding_3(self):
        s1, s2, s3, s4 = z3.Ints('s1 s2 s3 s4')

        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super(BasicBlock, self).__init__()

            def forward(self, x: TensorType([Dyn, 1]), y: TensorType([Dyn])):
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        # print(transformed)
        assert s.check() == z3.sat

        x = z3.Const(1, tensor_type)
        y = z3.Const(2, tensor_type)

        s.add(s2 != 0)
        s.add(x == tensor_type.tensor2(D(0, s1), D(s2, 1)))
        s.add(y == tensor_type.tensor1(D(0, s3)))

        assert s.check() == z3.sat

        # print(s.model())

        add_result = z3.Const(3, tensor_type)
        assert s.model()[add_result].arg(0).arg(0).as_long() == 0
        assert s.model()[add_result].arg(1).arg(0).as_long() == 0


    def test_add_padding_4(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super(BasicBlock, self).__init__()

            def forward(self, x: TensorType([2, 1]), y: TensorType([3])):
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)

        s = z3.Solver()
        s.add(transformed)

        assert s.check() == z3.sat

        add_result = z3.Const(3, tensor_type)
        assert s.model()[add_result] == tensor_type.tensor2(D(1, 2), D(1, 3))

    def test_add_padding_5(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super(BasicBlock, self).__init__()

            def forward(self, x: TensorType([2, 2]), y: TensorType([3])):
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        assert s.check() == z3.unsat

    def test_add_size_3(self):

        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super(BasicBlock, self).__init__()

            def forward(self, x: TensorType([Dyn, Dyn, Dyn]), y: TensorType([Dyn, Dyn, Dyn])):
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        assert s.check() == z3.sat

        x = z3.Const(1, tensor_type)
        y = z3.Const(2, tensor_type)

        s1, s2, s3, s4, s5 = z3.Ints('s1 s2 s3 s4 s5')

        s.add(x == tensor_type.tensor3(D(1, s1), D(1, 1), D(1, s2)))
        s.add(y == tensor_type.tensor3(D(1, s3), D(1, s4), D(1, s5)))

        assert s.check() == z3.sat
        s.add(s2 == 5)
        assert s.check() == z3.sat
        s.add(s5 == 6)
        assert s.check() == z3.unsat

    def test_add_padding_6(self):

        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super(BasicBlock, self).__init__()

            def forward(self, x: TensorType([Dyn]), y: TensorType([Dyn, Dyn, Dyn])):
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        assert s.check() == z3.sat

        x = z3.Const(1, tensor_type)
        y = z3.Const(2, tensor_type)

        s1, s2, s3, s4, s5 = z3.Ints('s1 s2 s3 s4 s5')

        s.add(x == tensor_type.tensor1(D(1, s1)))
        s.add(y == tensor_type.tensor3(D(1, s2), D(1, s3), D(1, s4)))

        assert s.check() == z3.sat

        s.add(s1 == 4)
        s.add(s4 == 5)

        assert s.check() == z3.unsat

    def test_add_padding_7(self):

        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super(BasicBlock, self).__init__()

            def forward(self, x: TensorType([Dyn]), y: TensorType([Dyn, Dyn, Dyn, Dyn])):
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        assert s.check() == z3.sat
        x = z3.Const(1, tensor_type)
        s1, s2, s3, s4, s5 = z3.Ints('s1 s2 s3 s4 s5')
        s.add(x == tensor_type.tensor2(D(s1, s2), D(s2, s3)))
        assert s.check() == z3.unsat


    def test_add_padding_8(self):

        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super(BasicBlock, self).__init__()

            def forward(self, x: TensorType([Dyn]), y: TensorType([Dyn, Dyn, Dyn, Dyn])):
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        assert s.check() == z3.sat
        x = z3.Const(1, tensor_type)
        y = z3.Const(2, tensor_type)

        s1, s2, s3, s4, s5 = z3.Ints('s1 s2 s3 s4 s5')
        s.add(x == tensor_type.tensor1(D(s1, 1)))
        s.add(s1 >= 0)

        assert s.check() == z3.sat

        s.add(y == tensor_type.tensor4(D(0, s2), D(0, s3), D(0, s4), D(0, s5)))
        assert s.check() == z3.sat

    def test_add_padding_9(self):

        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super(BasicBlock, self).__init__()

            def forward(self, x: Dyn, y: TensorType([Dyn, Dyn, Dyn, Dyn])):
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)

        assert s.check() == z3.sat
        x = z3.Const(1, tensor_type)
        y = z3.Const(2, tensor_type)

        s1, s2, s3, s4, s5, s6, s7 = z3.Ints('s1 s2 s3 s4 s5 s6 s7')
        s.add(x == tensor_type.tensor1(D(s1, s7)))
        s.add(s1 == 1)
        assert s.check() == z3.sat

        s.add(y == tensor_type.tensor4(D(0, s2), D(0, s3), D(0, s4), D(s6, s5)))
        assert s.check() == z3.sat

        s.add(s6 == 1)

        assert s.check() == z3.sat
        s.add(s5 != 1, s7 != 1)
        assert s.check()

        assert s.model()[s5].as_long() == s.model()[s7].as_long()

    def test_conv_static(self):
        s1, s2, s3, s4 = z3.Ints('s1 s2 s3 s4')
        e1, e2, e3, e4 = z3.Ints('e1 e2 e3 e4')
        s11, s22, s33, s44 = z3.Ints('s11 s22 s33 s44')
        e11, e22, e33, e44 = z3.Ints('e11 e22 e33 e44')
        d1, d2, d3, d4 = D(s11, s1), D(s22, s2), D(s33, s3), D(s44, s4),
        b1, b2, b3, b4 = D(e11, e1), D(e22, e2), D(e33, e3), D(e44, e4)

        class BasicBlock(torch.nn.Module):
            def __init__(self, in_planes, out_planes, kernel_size, stride, padding, groups, dilation):
                super(BasicBlock, self).__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                                             kernel_size=kernel_size, stride=stride,
                                             padding=padding, dilation=dilation)

            def forward(self, x: TensorType((1, 2, 10, 20))):
                return self.conv1(x)

        ast_rewriter = RewritingTracer()

        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)
        res = B.forward(torch.rand(1, 2, 10, 20)).size()

        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        new_transformed_c = transform_all_constraints(traced)
        solver = z3.Solver()
        solver.add(new_transformed_c)
        assert solver.check() == z3.sat

        x = z3.Const(1, tensor_type)
        y = z3.Const(2, tensor_type)

        solver.add(x == tensor_type.tensor4(d1, d2, d3, d4))
        solver.add(y == tensor_type.tensor4(b1, b2, b3, b4))
        assert solver.check() == z3.sat
        # print(solver.model())
        assert solver.model()[e3].as_long() == res[2]
        assert solver.model()[e4].as_long() == res[3]

        B2 = BasicBlock(2, 4, 5, 2, 9, 2, 2)
        res2 = B2.forward(torch.rand(1, 2, 10, 20)).size()

        graph2 = ast_rewriter.trace(B2)
        traced2 = GraphModule(ast_rewriter.root, graph2, "gm")
        new_transformed_c = transform_all_constraints(traced2)
        solver = z3.Solver()
        solver.add(new_transformed_c)

        solver.add(x == tensor_type.tensor4(d1, d2, d3, d4))
        solver.add(y == tensor_type.tensor4(b1, b2, b3, b4))

        assert solver.check() == z3.sat
        assert solver.model()[e3].as_long() == res2[2]
        assert solver.model()[e4].as_long() == res2[3]

    def test_reshape_dyn(self):
        s11, s22, s33, s44 = z3.Ints('s11 s22 s33 s44')

        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super(BasicBlock, self).__init__()

            def forward(self, x: Dyn):
                return torch.reshape(x, (2, -1))

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced)
        s = z3.Solver()
        s.add(transformed)
        assert s.check() == z3.sat
        x = z3.Const(1, tensor_type)
        s.add(x == tensor_type.tensor1(D(1, s11)))
        assert s.check() == z3.sat
        s.add(z3.Or([s11 == 2, s11 == 4, s11 == 9]))
        assert s.check() == z3.sat
        s.add(s11 == 9)
        assert s.check() == z3.unsat


    def test_reshape_annotated(self):
        s1, s2, s3, s4 = z3.Ints('s1 s2 s3 s4')
        s11, s22, s33, s44 = z3.Ints('s11 s22 s33 s44')
        d1, d2, d3, d4 = D(s11, s1), D(s22, s2), D(s33, s3), D(s44, s4),

        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super(BasicBlock, self).__init__()

            def forward(self, x: TensorType([Dyn])):
                return torch.reshape(x, (2, -1))

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        transformed = transform_all_constraints(traced)
        s = z3.Solver()
        s.add(transformed)
        assert s.check() == z3.sat
        x = z3.Const(1, tensor_type)
        s.add(x == tensor_type.tensor2(d1, d2))
        assert s.check() == z3.unsat

    def test_reshape_static_target(self):
        s11, s22, s33, s44 = z3.Ints('s11 s22 s33 s44')

        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super(BasicBlock, self).__init__()

            def forward(self, x: TensorType([Dyn])):
                return torch.reshape(x, (2, 3))

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        transformed = transform_all_constraints(traced)
        # print(transformed)
        s = z3.Solver()
        s.add(transformed)
        assert s.check() == z3.sat
        x = z3.Const(1, tensor_type)
        s.add(x == tensor_type.tensor1(D(1, s11)))
        s.check()
        assert s.model()[s11].as_long() == 6
        s.add(s11 != 6)
        assert s.check() == z3.unsat

    def test_reshape_static_target2(self):
        s11, s22, s33, s44 = z3.Ints('s11 s22 s33 s44')

        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super(BasicBlock, self).__init__()

            def forward(self, x: Dyn):
                return torch.reshape(x, (2, 3, 1, 1))

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        transformed = transform_all_constraints(traced)
        s = z3.Solver()
        s.add(transformed)
        assert s.check() == z3.sat
        x = z3.Const(1, tensor_type)
        s.add(x == tensor_type.tensor1(D(1, s11)))
        s.check()
        assert s.model()[s11].as_long() == 6
        s.add(s11 != 6)
        assert s.check() == z3.unsat

    def test_conv2D_maxpool2d_flatten(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super(BasicBlock, self).__init__()

                self.conv1 = torch.nn.Conv2d(3, 6, 5)
                self.pool = torch.nn.MaxPool2d(2, 2)
                self.conv2 = torch.nn.Conv2d(6, 16, 5)
                self.fc1 = torch.nn.Linear(5, 120)
                self.pool2 = torch.nn.AdaptiveAvgPool2d((6, 7))

            def forward(self, x : TensorType((4, 3, 32, 32))):
                out = self.conv1(x)
                out = self.pool(out)
                out = self.conv2(out)
                out = self.pool(out)
                out = self.fc1(out)
                out = self.pool2(out)
                out = torch.flatten(out, 1)
                return out

        B = BasicBlock()
        res = B.forward(torch.rand(4, 3, 32, 32)).shape
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        constraints = transform_all_constraints(traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        solver.check()
        input = z3.Const(1, tensor_type)
        solver.add(input == tensor_type.tensor4(D(1, 4), D(1, 3), D(1, 32), D(1, 32)))
        solver.check()
        output = z3.Const(48, tensor_type)
        assert solver.model()[output].arg(0).arg(1) == res[0]
        assert solver.model()[output].arg(1).arg(1) == res[1]

    def test_conv2D_maxpool2d_flatten_unsat(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super(BasicBlock, self).__init__()

                self.conv1 = torch.nn.Conv2d(3, 6, 5)
                self.pool = torch.nn.MaxPool2d(2, 2)
                self.conv2 = torch.nn.Conv2d(6, 16, 5)
                self.fc1 = torch.nn.Linear(5, 120)
                self.pool2 = torch.nn.AdaptiveAvgPool2d((6, 7))

            def forward(self, x : TensorType((4, 3, 32, 32))):
                out = self.conv1(x)
                out = self.pool(out)
                out = self.conv2(out)
                out = self.pool(out)
                out = self.fc1(out)
                out = self.pool2(out)
                out = torch.flatten(out, 1)
                return out

        B = BasicBlock()
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        constraints = transform_all_constraints(traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        solver.check()
        input = z3.Const(1, tensor_type)
        solver.add(input == tensor_type.tensor4(D(1, 4), D(1, 3), D(1, 32), D(1, 45)))
        assert solver.check() == z3.unsat

    def test_conv2D_maxpool2d_flatten_dyn(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super(BasicBlock, self).__init__()

                self.conv1 = torch.nn.Conv2d(3, 6, 5)
                self.pool = torch.nn.MaxPool2d(2, 2)
                self.conv2 = torch.nn.Conv2d(6, 16, 5)
                self.fc1 = torch.nn.Linear(5, 120)
                self.pool2 = torch.nn.AdaptiveAvgPool2d((6, 7))

            def forward(self, x : TensorType((Dyn, 3, 32, 32))):
                out = self.conv1(x)
                out = self.pool(out)
                out = self.conv2(out)
                out = self.pool(out)
                out = self.fc1(out)
                out = self.pool2(out)
                out = torch.flatten(out, 1)
                return out

        B = BasicBlock()
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        constraints = transform_all_constraints(traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        assert solver.check() == z3.sat

    def test_type_check_flatten(self):
        s1, s2, s3, s4 = z3.Ints('s1 s2 s3 s4')

        class M(torch.nn.Module):
            def forward(self, x: TensorType([2, 3, 4, 5])):
                return torch.flatten(x, start_dim=1, end_dim=3)

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        constraints = transform_all_constraints(symbolic_traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        assert solver.check() == z3.sat
        flatten = z3.Const(2, tensor_type)

        res = M().forward(torch.rand(2, 3, 4, 5)).size()
        assert solver.model()[flatten].arg(0).arg(1) == res[0]
        assert solver.model()[flatten].arg(1).arg(1) == res[1]

        class M(torch.nn.Module):
            def forward(self, x: TensorType([2, 3, Dyn, 5])):
                return torch.flatten(x, start_dim=1, end_dim=3)

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        constraints = transform_all_constraints(symbolic_traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        assert solver.check() == z3.sat
        x = z3.Const(1, tensor_type)
        y = z3.Const(2, tensor_type)

        solver.add(x == tensor_type.tensor4(D(1, 2), D(1, 3), D(0, s1), D(1, 5)))
        assert solver.check() == z3.sat
        assert solver.model()[y].arg(1).arg(0) == 0


        class M(torch.nn.Module):
            def forward(self, x: TensorType([2, 3, Dyn])):
                return torch.flatten(x, 10, 0)

        module = M()
        # print(module.forward(torch.rand(2,3,5)).shape)
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        constraints = transform_all_constraints(symbolic_traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        assert solver.check() == z3.unsat

class ConstraintGeneration(unittest.TestCase):

    def test_add_reshape(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super(BasicBlock, self).__init__()

            def forward(self, x: Dyn, y: Dyn):
                return torch.add(torch.reshape(x, (1, 2)), torch.reshape(y, (2, 2)))

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        generator = ConstraintGenerator(traced)
        new_constraints, counter = generator.generate_constraints(0)
        assert len(new_constraints.conjucts) == 11


    def test_conv_reshape_add(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self, in_planes, out_planes, kernel_size, stride, padding, groups, dilation):
                super(BasicBlock, self).__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                                             kernel_size=kernel_size, stride=stride,
                                             padding=padding, groups=groups, bias=False, dilation=dilation)

            def forward(self, x: Dyn, y: Dyn):
                return torch.add(self.conv1(torch.reshape(x, (1, 2, 10, 20))), y)

        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        generator = ConstraintGenerator(traced)
        new_constraints, counter = generator.generate_constraints(0)
        assert len(new_constraints.conjucts) == 16


class TestInternalConstraints(unittest.TestCase):
    def test_precision(self):

        c1 = BinConstraintT(Dyn, TVar('x'), op_precision)
        transformed, _ = transform_constraint(c1, 0)
        assert transformed == T()

        c2 = BinConstraintT(TensorType([1, Dyn, 3]), TVar('x'), op_precision)
        transformed, counter = transform_constraint(c2, 0)
        assert len(transformed.conjucts) == 7

    def test_matching(self):
        c1 = BinConstraintT(TVar('x'),
                            TensorType([DVar('a'), DVar('b'), DVar('c'), DVar('d')]), op_matching)
        transformed, _ = transform_constraint(c1, 0)
        assert len(transformed.disjuncts) == 2

    def test_consistency(self):
        c1 = BinConstraintT(TVar('x'),
                            TensorType([DVar('a'), DVar('b')]), op_consistency)
        transformed, count = transform_constraint(c1, 0)

        assert len(transformed.disjuncts) == 5
        transformed, count = transform_constraint(transformed, count)
        assert len(transformed.disjuncts) == 5

    def test_apply_broadcasting(self):
        c1 = ApplyBroadcasting(TVar(1), TVar(2), TVar(3), TVar(4))
        transformed, count = transform_apply_broadcasting(c1, 5)
        assert len(transformed.conjucts) == 41

@skipIfNoTorchVision
class TestResNet(unittest.TestCase):

    def test_resnet50_unsat(self):
        traced = symbolic_trace(models.resnet50())
        for n in traced.graph.nodes:
            n.type = Dyn

        constraints = transform_all_constraints(traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        input = z3.Const(1, tensor_type)
        # input with 3 dimensions
        solver.add(input == tensor_type.tensor3(D(1, 1), D(1, 3), D(1, 224)))
        assert solver.check() == z3.unsat



    def test_resnet50(self):
        traced = symbolic_trace(models.resnet50())
        for n in traced.graph.nodes:
            n.type = Dyn

        sample_input = torch.randn(1, 3, 224, 224)
        res = models.resnet50().forward(sample_input).size()
        constraints = transform_all_constraints(traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        assert solver.check() == z3.sat
        linear = z3.Const(650, tensor_type)

        input = z3.Const(1, tensor_type)
        solver.add(input == tensor_type.tensor4(D(1, 1), D(1, 3), D(1, 224), D(1, 224)))
        assert solver.check() == z3.sat
        assert solver.model()[linear] == tensor_type.tensor2(D(1, res[0]), D(1, res[1]))

    def test_resnet502(self):
        traced = symbolic_trace(models.resnet50())
        for n in traced.graph.nodes:
            n.type = Dyn

        constraints = transform_all_constraints(traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        linear = z3.Const(650, tensor_type)
        input = z3.Const(1, tensor_type)
        batch = z3.Int('b')
        solver.add(input == tensor_type.tensor4(D(1, batch), D(1, 3), D(1, 224), D(1, 224)))
        solver.add(batch > 4)
        solver.check()
        assert solver.model()[batch] == solver.model()[linear].arg(0).arg(1)

    def test_resnet503(self):
        traced = symbolic_trace(models.resnet50())
        for n in traced.graph.nodes:
            n.type = Dyn

        constraints = transform_all_constraints(traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        linear = z3.Const(650, tensor_type)
        input = z3.Const(1, tensor_type)
        batch, d1, d2 = z3.Ints('b d1 d2')
        solver.add(input == tensor_type.tensor4(D(1, batch), D(1, 3), D(1, 224), D(1, 224)))
        solver.add(linear == tensor_type.tensor2(D(1, d1), D(1, d2)))
        assert solver.check() == z3.sat
        solver.add(batch != d1)
        assert solver.check() == z3.unsat

@skipIfNoTorchVision
class TestAlexNet(unittest.TestCase):
    def test_alexnet1(self):

        alexnet = models.alexnet()
        symbolic_traced : torch.fx.GraphModule = symbolic_trace(alexnet)

        for n in symbolic_traced.graph.nodes:
            n.type = Dyn

        # print(symbolic_traced)

        res = alexnet.forward(torch.rand(10, 3, 227, 227)).size()
        constraints = transform_all_constraints(symbolic_traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        assert solver.check() == z3.sat
        input = z3.Const(1, tensor_type)
        conv = z3.Const(2, tensor_type)
        solver.add(input == tensor_type.tensor4(D(1, 10), D(1, 3), D(1, 227), D(1, 227)))
        assert solver.check() == z3.sat
        assert solver.model()[conv] == tensor_type.tensor4(D(1, 10), D(1, 64), D(1, 56), D(1, 56))

        relu = z3.Const(7, tensor_type)
        assert solver.model()[relu] == tensor_type.tensor4(D(1, 10), D(1, 64), D(1, 56), D(1, 56))

        maxpool = z3.Const(8, tensor_type)
        assert solver.model()[maxpool] == tensor_type.tensor4(D(1, 10), D(1, 64), D(1, 27), D(1, 27))

        maxpool2 = z3.Const(42, tensor_type)
        assert solver.model()[maxpool2] == tensor_type.tensor4(D(1, 10), D(1, 256), D(1, 6), D(1, 6))

        flatten = z3.Const(52, tensor_type)
        assert solver.model()[flatten] == tensor_type.tensor2(D(1, 10), D(1, 9216))

        linear = z3.Const(64, tensor_type)
        assert solver.model()[linear] == tensor_type.tensor2(D(1, 10), D(1, 4096))

        linear2 = z3.Const(109, tensor_type)
        assert solver.model()[linear2] == tensor_type.tensor2(D(1, res[0]), D(1, res[1]))


    def test_alexnet2(self):
        alexnet = models.alexnet()
        symbolic_traced : torch.fx.GraphModule = symbolic_trace(alexnet)

        for n in symbolic_traced.graph.nodes:
            if n.op == 'placeholder':
                n.type = TensorType([Dyn, 4, 227, 227])

        constraints = transform_all_constraints(symbolic_traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        assert solver.check() == z3.unsat

    def test_alexnet3(self):
        alexnet = models.alexnet()
        symbolic_traced : torch.fx.GraphModule = symbolic_trace(alexnet)

        for n in symbolic_traced.graph.nodes:
            if n.op == 'placeholder':
                n.type = TensorType([Dyn, Dyn, 227, 227])

        constraints = transform_all_constraints(symbolic_traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        assert solver.check() == z3.sat

    def test_alexnet4(self):
        alexnet = models.alexnet()
        symbolic_traced : torch.fx.GraphModule = symbolic_trace(alexnet)

        for n in symbolic_traced.graph.nodes:
            if n.op == 'placeholder':
                n.type = TensorType([Dyn, Dyn, 227])

        constraints = transform_all_constraints(symbolic_traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        assert solver.check() == z3.unsat



if __name__ == '__main__':
    unittest.main()
