# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Example code to do square matrix multiplication."""

import tvm
import numpy as np
import tvm.testing
from tvm.script import tir as T


_dtype = "float32"


def write_code(code, fname):
    with open(fname, "w") as f:
        f.write(code)


M = N = K = 1024


# @tvm.script.ir_module
# class MyModule:
#     @T.prim_func
#     def main(a: T.handle, b: T.handle, c: T.handle):
#         T.func_attr({"global_symbol": "main", "tir.noalias": True})
#         A = T.match_buffer(a, [M, K])
#         B = T.match_buffer(b, [K, N])
#         C = T.match_buffer(c, [M, N])

#         for i in range(M):
#             for j in range(N):
#                 for k in range(K):
#                     with T.block("B"):
#                         vi, vj, vk = T.axis.remap("SSR", [i, j, k])
#                         with T.init():
#                             C[vi, vj] = 0.0
#                         C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M, K])
        B = T.match_buffer(b, [K, N])
        C = T.match_buffer(c, [M, N])

        for i, j, k in T.grid(M, N, K):
            with T.block("B"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]



ir_module = MyModule
sch = tvm.tir.Schedule(ir_module, debug_mask="all")

print(type(ir_module))
print(ir_module.script())

write_code(sch.mod.astext(), "0.original.cu")
block_b = sch.get_block("B")
(i, j, k) = sch.get_loops(block_b)
sch.bind(i, "blockIdx.x")
sch.bind(j, "threadIdx.x")


write_code(sch.mod.astext(), "1.thread_bind.cu")

ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")

write_code(cuda_mod.imported_modules[0].get_source(), "tmp.cu")

cuda_a = tvm.nd.array(np.arange(M * K).reshape((M, K)).astype(_dtype), ctx)
cuda_b = tvm.nd.array(np.arange(K * N).reshape((K, N)).astype(_dtype), ctx)
cuda_c = tvm.nd.array(np.zeros((M, N)).astype(_dtype), ctx)
cuda_mod(cuda_a, cuda_b, cuda_c)

num_flops = 2 * M * K * N
num_runs = 10
timer_cuda_mod  = cuda_mod.time_evaluator(
    cuda_mod.entry_name, ctx, number=num_runs)

t = timer_cuda_mod(cuda_a, cuda_b, cuda_c).mean

GFLOPS = num_flops / (t * 1e3) / 1e6
print("average time cost of %d runs = %g ms, %g GFLOPS." %
      (num_runs, t * 1e3, GFLOPS))
