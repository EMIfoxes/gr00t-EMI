import torch
import warnings

def test_flash_attn():
    """测试 Flash Attention 是否可用"""
    
    print("=" * 60)
    print("Flash Attention 可用性测试")
    print("=" * 60)
    
    # 1. 检查 PyTorch 版本和 CUDA 可用性
    print(f"\n1. PyTorch 版本: {torch.__version__}")
    print(f"   CUDA 可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA 版本: {torch.version.cuda}")
        print(f"   GPU 设备: {torch.cuda.get_device_name(0)}")
    
    # 2. 尝试导入 flash_attn
    print("\n2. 测试 flash_attn 导入...")
    try:
        import flash_attn
        print(f"   ✅ flash_attn 已安装 (版本: {flash_attn.__version__})")
    except ImportError:
        print("   ❌ flash_attn 未安装")
        print("   安装命令: pip install flash-attn --no-build-isolation")
        return False
    
    # 3. 尝试导入核心函数
    print("\n3. 测试核心函数导入...")
    try:
        from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
        print("   ✅ flash_attn_func 导入成功")
        print("   ✅ flash_attn_qkvpacked_func 导入成功")
    except ImportError as e:
        print(f"   ❌ 导入失败: {e}")
        return False
    
    # 4. 功能性测试（需要 CUDA）
    print("\n4. 功能性测试...")
    if not torch.cuda.is_available():
        print("   ⚠️  跳过: 需要 CUDA GPU")
        return True  # 包已安装但无法测试功能
    
    try:
        # 设置测试参数
        batch_size = 2
        seqlen = 128
        nheads = 8
        headdim = 64
        
        # 创建随机输入 (需要在 GPU 上)
        device = "cuda"
        dtype = torch.float16  # Flash Attention 通常需要 fp16 或 bf16
        
        print(f"   测试配置: batch={batch_size}, seqlen={seqlen}, heads={nheads}, dim={headdim}")
        
        # 创建 Q, K, V 张量
        q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
        
        # 运行 Flash Attention
        with torch.cuda.amp.autocast(dtype=dtype):
            output = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
        
        print(f"   ✅ 前向传播成功")
        print(f"   输出形状: {output.shape}")
        print(f"   输出设备: {output.device}")
        print(f"   输出类型: {output.dtype}")
        
        # 测试反向传播
        output.sum().backward()
        print("   ✅ 反向传播成功")
        
        # 5. 测试因果掩码功能
        print("\n5. 测试因果掩码...")
        with torch.cuda.amp.autocast(dtype=dtype):
            output_causal = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
        print("   ✅ 因果掩码模式成功")
        
        # 6. 测试变长序列功能（可选）
        print("\n6. 测试变长序列支持...")
        try:
            from flash_attn import flash_attn_varlen_func
            print("   ✅ flash_attn_varlen_func 可用")
        except ImportError:
            print("   ⚠️  flash_attn_varlen_func 不可用")
        
    except Exception as e:
        print(f"   ❌ 功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！Flash Attention 可正常使用")
    print("=" * 60)
    return True

def benchmark_flash_attn():
    """简单的性能对比测试"""
    print("\n" + "=" * 60)
    print("性能对比测试 (Flash Attention vs 标准 Attention)")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("需要 CUDA GPU 进行性能测试")
        return
    
    try:
        from flash_attn import flash_attn_func
        import time
        
        batch_size = 8
        seqlen = 1024
        nheads = 16
        headdim = 64
        
        device = "cuda"
        dtype = torch.float16
        
        q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
        
        # 预热
        for _ in range(10):
            _ = flash_attn_func(q, k, v, causal=True)
        torch.cuda.synchronize()
        
        # 测试 Flash Attention
        n_iters = 100
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(n_iters):
            out_flash = flash_attn_func(q, k, v, causal=True)
        end.record()
        torch.cuda.synchronize()
        flash_time = start.elapsed_time(end) / n_iters
        
        print(f"Flash Attention: {flash_time:.3f} ms/iter")
        
        # 测试标准 Attention (如果内存允许)
        try:
            q_f = q.transpose(1, 2)  # [batch, heads, seqlen, dim]
            k_f = k.transpose(1, 2)
            v_f = v.transpose(1, 2)
            
            start.record()
            for _ in range(n_iters):
                scores = torch.matmul(q_f, k_f.transpose(-2, -1)) / (headdim ** 0.5)
                scores = torch.softmax(scores, dim=-1)
                out_std = torch.matmul(scores, v_f)
            end.record()
            torch.cuda.synchronize()
            std_time = start.elapsed_time(end) / n_iters
            
            print(f"标准 Attention: {std_time:.3f} ms/iter")
            print(f"加速比: {std_time/flash_time:.2f}x")
        except Exception as e:
            print(f"标准 Attention 测试跳过: {e}")
            
    except Exception as e:
        print(f"性能测试失败: {e}")

if __name__ == "__main__":
    success = test_flash_attn()
    if success and torch.cuda.is_available():
        benchmark_flash_attn()