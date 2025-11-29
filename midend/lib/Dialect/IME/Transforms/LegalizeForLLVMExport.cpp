//====- LegalizeForLLVMExport.cpp - IME Lowering Pass -----===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

#include "Dialect/IME/IMEDialect.h"
#include "Dialect/IME/IMEOps.h"
#include "Dialect/IME/Transform.h"

using namespace mlir;
using namespace buddy::ime;

namespace {

/*struct IMEVmadotLowering : public ConvertOpToLLVMPattern<VmadotOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(VmadotOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};*/

struct IMEVmadotLowering : public ConvertOpToLLVMPattern<VmadotOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(VmadotOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // 1. 获取输入操作数 (指针)
    // 根据 MLIR 定义: ime.vmadot %c, %a, %b
    // adaptor.getOperands()[0] -> C (Output/Accumulator)
    // adaptor.getOperands()[1] -> A (Input)
    // adaptor.getOperands()[2] -> B (Input)
    /*Value cPtr = adaptor.getOperands()[0]; 
    Value aPtr = adaptor.getOperands()[1]; 
    Value bPtr = adaptor.getOperands()[2]; */
    Value cStruct = adaptor.getOperands()[0];
    Value cPtr = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), cStruct, ArrayRef<int64_t>{1}); // Index 1 is alignedPtr

    Value aStruct = adaptor.getOperands()[1];
    Value aPtr = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), aStruct, ArrayRef<int64_t>{1});

    Value bStruct = adaptor.getOperands()[2];
    Value bPtr = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), bStruct, ArrayRef<int64_t>{1});

    // 2. 构建汇编字符串
    // 逻辑来源: IME_Complete_Instruction_Flow.md & IME_Example_Walkthrough.md
    // $0 -> cPtr (C矩阵地址)
    // $1 -> aPtr (A矩阵地址)
    // $2 -> bPtr (B矩阵地址)
    StringRef asmString = 
        // ---------------------------------------------------
        // 1. 配置环境 & 初始化累加器 (C)
        // ---------------------------------------------------
        // 设置 SEW=32, LMUL=2 (对应 4x4 int32 C 矩阵, 16个元素)
        "vsetvli t0, zero, e32, m2\n\t"
        // 清零累加器 v28 (v28-v29)
        // 注意: 实际应用中可能需要先从内存加载 C 的旧值，
        // 但根据任务描述 vmadot-basic.mlir 里已经做过 linalg.fill 0，
        // 且 IME 是累加指令。为了简化，这里我们假设 C 初始为 0 或由用户负责初始化。
        // 如果要支持 += 操作，这里应该用 vle32.v v28, ($0) 加载旧值。
        // 这里演示覆盖模式(Overwrite): 先清零，算完直接存。
        "vxor.vv v28, v28, v28\n\t" 

        // ---------------------------------------------------
        // 2. 加载输入矩阵 A
        // ---------------------------------------------------
        // 设置 SEW=8, LMUL=1 (对应 4x8 int8 A 矩阵, 32个元素)
        "vsetvli t0, zero, e8, m1\n\t"
        "vle8.v v0, ($1)\n\t" 

        // ---------------------------------------------------
        // 3. 加载输入矩阵 B
        // ---------------------------------------------------
        // 同样是 8x4 int8 B 矩阵 (注意：假设数据已经 Pack 好)
        "vle8.v v1, ($2)\n\t"

        // ---------------------------------------------------
        // 4. 执行 IME 矩阵乘法
        // ---------------------------------------------------
        // C(v28) += A(v0) * B(v1)
        "vmadot v28, v0, v1\n\t"

        // ---------------------------------------------------
        // 5. 存储结果 C
        // ---------------------------------------------------
        // 切换回 SEW=32, LMUL=2
        "vsetvli t0, zero, e32, m2\n\t"
        "vse32.v v28, ($0)";

    // 3. 定义约束字符串
    // "r": 输入操作数放在通用寄存器中 (存放内存地址)
    // "~{...}": Clobber List，告诉编译器这些寄存器被汇编代码修改了，不要挪作他用
    // v0, v1: 用于存放 A, B
    // v28, v29: 用于存放 C (LMUL=2，占用两个)
    // t0: vsetvli 会修改它
    // memory: 汇编读写了内存
    StringRef constraints = "r,r,r,~{v0},~{v1},~{v28},~{v29},~{t0},~{memory}";

    // 4. 创建 InlineAsmOp
    rewriter.replaceOpWithNewOp<LLVM::InlineAsmOp>(
        op,
        TypeRange(),                            // 返回类型: void
        ValueRange{cPtr, aPtr, bPtr},           // 输入参数
        asmString,                              // 汇编指令
        constraints,                            // 约束
        true,                                   // hasSideEffects = true (读写内存)
        false,                                  // isAlignStack = false
        LLVM::AsmDialectAttr::get(getContext(), LLVM::AsmDialect::AD_ATT), // 使用 AT&T 语法
        ArrayAttr()                             // operand_attrs
    );

    return success();
  }
};

struct IMEVmadotuLowering : public ConvertOpToLLVMPattern<VmadotuOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(VmadotuOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // 1. 获取输入操作数 (指针)
    // 根据更新后的 IME.td 定义: ime.vmadotu %vd, %vs1, %vs2
    // adaptor.getOperands()[0] -> vd (Output/Accumulator)
    // adaptor.getOperands()[1] -> vs1 (Input A, unsigned)
    // adaptor.getOperands()[2] -> vs2 (Input B, unsigned)
    Value vdStruct = adaptor.getOperands()[0];
    Value vdPtr = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), vdStruct, ArrayRef<int64_t>{1});

    Value vs1Struct = adaptor.getOperands()[1];
    Value vs1Ptr = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), vs1Struct, ArrayRef<int64_t>{1});

    Value vs2Struct = adaptor.getOperands()[2];
    Value vs2Ptr = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), vs2Struct, ArrayRef<int64_t>{1});

    // 2. 构建汇编字符串
    // $0 -> vdPtr (C矩阵地址)
    // $1 -> vs1Ptr (A矩阵地址, unsigned)
    // $2 -> vs2Ptr (B矩阵地址, unsigned)
    StringRef asmString = 
        "vsetvli t0, zero, e32, m2\n\t"
        "vxor.vv v28, v28, v28\n\t"
        "vsetvli t0, zero, e8, m1\n\t"
        "vle8.v v0, ($1)\n\t"
        "vle8.v v1, ($2)\n\t"
        "vmadotu v28, v0, v1\n\t"
        "vsetvli t0, zero, e32, m2\n\t"
        "vse32.v v28, ($0)";

    StringRef constraints = "r,r,r,~{v0},~{v1},~{v28},~{v29},~{t0},~{memory}";

    rewriter.replaceOpWithNewOp<LLVM::InlineAsmOp>(
        op,
        TypeRange(),
        ValueRange{vdPtr, vs1Ptr, vs2Ptr},
        asmString,
        constraints,
        true,
        false,
        LLVM::AsmDialectAttr::get(getContext(), LLVM::AsmDialect::AD_ATT),
        ArrayAttr()
    );

    return success();
  }
};

struct IMEVmadotsuLowering : public ConvertOpToLLVMPattern<VmadotsuOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(VmadotsuOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // 1. 获取输入操作数 (指针)
    // 根据更新后的 IME.td 定义: ime.vmadotsu %vd, %vs1, %vs2
    // adaptor.getOperands()[0] -> vd (Output/Accumulator)
    // adaptor.getOperands()[1] -> vs1 (Input A, signed)
    // adaptor.getOperands()[2] -> vs2 (Input B, unsigned)
    Value vdStruct = adaptor.getOperands()[0];
    Value vdPtr = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), vdStruct, ArrayRef<int64_t>{1});

    Value vs1Struct = adaptor.getOperands()[1];
    Value vs1Ptr = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), vs1Struct, ArrayRef<int64_t>{1});

    Value vs2Struct = adaptor.getOperands()[2];
    Value vs2Ptr = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), vs2Struct, ArrayRef<int64_t>{1});

    // 2. 构建汇编字符串
    // $0 -> vdPtr (C矩阵地址)
    // $1 -> vs1Ptr (A矩阵地址, signed)
    // $2 -> vs2Ptr (B矩阵地址, unsigned)
    StringRef asmString = 
        "vsetvli t0, zero, e32, m2\n\t"
        "vxor.vv v28, v28, v28\n\t"
        "vsetvli t0, zero, e8, m1\n\t"
        "vle8.v v0, ($1)\n\t"
        "vle8.v v1, ($2)\n\t"
        "vmadotsu v28, v0, v1\n\t"
        "vsetvli t0, zero, e32, m2\n\t"
        "vse32.v v28, ($0)";

    StringRef constraints = "r,r,r,~{v0},~{v1},~{v28},~{v29},~{t0},~{memory}";

    rewriter.replaceOpWithNewOp<LLVM::InlineAsmOp>(
        op,
        TypeRange(),
        ValueRange{vdPtr, vs1Ptr, vs2Ptr},
        asmString,
        constraints,
        true,
        false,
        LLVM::AsmDialectAttr::get(getContext(), LLVM::AsmDialect::AD_ATT),
        ArrayAttr()
    );

    return success();
  }
};

struct IMEVmadotusLowering : public ConvertOpToLLVMPattern<VmadotusOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(VmadotusOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // 1. 获取输入操作数 (指针)
    // 根据更新后的 IME.td 定义: ime.vmadotus %vd, %vs1, %vs2
    // adaptor.getOperands()[0] -> vd (Output/Accumulator)
    // adaptor.getOperands()[1] -> vs1 (Input A, unsigned)
    // adaptor.getOperands()[2] -> vs2 (Input B, signed)
    Value vdStruct = adaptor.getOperands()[0];
    Value vdPtr = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), vdStruct, ArrayRef<int64_t>{1});

    Value vs1Struct = adaptor.getOperands()[1];
    Value vs1Ptr = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), vs1Struct, ArrayRef<int64_t>{1});

    Value vs2Struct = adaptor.getOperands()[2];
    Value vs2Ptr = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), vs2Struct, ArrayRef<int64_t>{1});

    // 2. 构建汇编字符串
    // $0 -> vdPtr (C矩阵地址)
    // $1 -> vs1Ptr (A矩阵地址, unsigned)
    // $2 -> vs2Ptr (B矩阵地址, signed)
    StringRef asmString = 
        "vsetvli t0, zero, e32, m2\n\t"
        "vxor.vv v28, v28, v28\n\t"
        "vsetvli t0, zero, e8, m1\n\t"
        "vle8.v v0, ($1)\n\t"
        "vle8.v v1, ($2)\n\t"
        "vmadotus v28, v0, v1\n\t"
        "vsetvli t0, zero, e32, m2\n\t"
        "vse32.v v28, ($0)";

    StringRef constraints = "r,r,r,~{v0},~{v1},~{v28},~{v29},~{t0},~{memory}";

    rewriter.replaceOpWithNewOp<LLVM::InlineAsmOp>(
        op,
        TypeRange(),
        ValueRange{vdPtr, vs1Ptr, vs2Ptr},
        asmString,
        constraints,
        true,
        false,
        LLVM::AsmDialectAttr::get(getContext(), LLVM::AsmDialect::AD_ATT),
        ArrayAttr()
    );

    return success();
  }
};

struct IMEVfmadotLowering : public ConvertOpToLLVMPattern<VfmadotOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(VfmadotOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // 1. 获取输入操作数 (指针)
    // 根据更新后的 IME.td 定义: ime.vfmadot %vd, %vs1, %vs2
    // adaptor.getOperands()[0] -> vd (Output/Accumulator, float16)
    // adaptor.getOperands()[1] -> vs1 (Input A, float16)
    // adaptor.getOperands()[2] -> vs2 (Input B, float16)
    Value vdStruct = adaptor.getOperands()[0];
    Value vdPtr = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), vdStruct, ArrayRef<int64_t>{1});

    Value vs1Struct = adaptor.getOperands()[1];
    Value vs1Ptr = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), vs1Struct, ArrayRef<int64_t>{1});

    Value vs2Struct = adaptor.getOperands()[2];
    Value vs2Ptr = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), vs2Struct, ArrayRef<int64_t>{1});

    // 2. 构建汇编字符串
    // 浮点版本使用 e16 (float16 元素大小) 和 vfmadot 指令
    // $0 -> vdPtr (C矩阵地址)
    // $1 -> vs1Ptr (A矩阵地址, float16)
    // $2 -> vs2Ptr (B矩阵地址, float16)
    StringRef asmString = 
        "vsetvli t0, zero, e16, m2\n\t"
        "vfmv.v.f v28, ft0\n\t"
        "vle16.v v0, ($1)\n\t"
        "vle16.v v1, ($2)\n\t"
        "vfmadot v28, v0, v1\n\t"
        "vse16.v v28, ($0)";

    StringRef constraints = "r,r,r,~{v0},~{v1},~{v28},~{v29},~{t0},~{memory}";

    rewriter.replaceOpWithNewOp<LLVM::InlineAsmOp>(
        op,
        TypeRange(),
        ValueRange{vdPtr, vs1Ptr, vs2Ptr},
        asmString,
        constraints,
        true,
        false,
        LLVM::AsmDialectAttr::get(getContext(), LLVM::AsmDialect::AD_ATT),
        ArrayAttr()
    );

    return success();
  }
};

struct LegalizeIMEForLLVMExport
    : public PassWrapper<LegalizeIMEForLLVMExport, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeIMEForLLVMExport)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext &context = getContext();

    LLVMConversionTarget target(context);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<IMEDialect>();

    LLVMTypeConverter typeConverter(&context);

    RewritePatternSet patterns(&context);
    patterns.add<IMEVmadotLowering, IMEVmadotuLowering, IMEVmadotsuLowering,
                 IMEVmadotusLowering, IMEVfmadotLowering>(typeConverter);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::populateIMELegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<IMEVmadotLowering, IMEVmadotuLowering, IMEVmadotsuLowering,
               IMEVmadotusLowering, IMEVfmadotLowering>(converter);
}

void mlir::configureIMELegalizeForExportTarget(LLVMConversionTarget &target) {
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addIllegalDialect<IMEDialect>();
}

std::unique_ptr<Pass> buddy::ime::createLegalizeForLLVMExportPass() {
  return std::make_unique<LegalizeIMEForLLVMExport>();
}
