import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPTokenizer
from PIL import Image
import numpy as np

class StableDiffusionDataset(Dataset):
    """
    Stable Diffusion LoRA 微调用的数据集类
    
    功能：
    1. 加载图像和文本
    2. 对图像进行预处理（resize、normalize）
    3. 对文本进行 tokenization
    4. 返回模型需要的格式
    """
    
    def __init__(
        self,
        dataset,  # HuggingFace 数据集
        tokenizer,  # CLIP tokenizer
        image_size=512,  # 图像大小
        conditioning_dropout_prob=0.1  # 条件dropout（用于无条件指导）
    ):
        """
        Args:
            dataset: HuggingFace datasets 对象
            tokenizer: transformers 的 CLIPTokenizer
            image_size: 训练用的图像大小（SD v1.5 一般用 512）
            conditioning_dropout_prob: 部分样本去掉文本条件的概率
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.conditioning_dropout_prob = conditioning_dropout_prob
        
        # 定义图像预处理流程
        self.image_transforms = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            # 中心裁剪，保证各向同性
            transforms.CenterCrop(image_size),
            # 转为 Tensor，范围 [0, 1]
            transforms.ToTensor(),
            # 标准化到 [-1, 1]（SD 的标准范围）
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        获取一个数据样本
        
        返回格式：
        {
            'pixel_values': Tensor,     # 预处理后的图像 [3, 512, 512]
            'input_ids': Tensor,        # Tokenized prompt [1, seq_len]
            'prompt': str,              # 原始 prompt（用于 text encoder）
        }
        """
        # 1️⃣ 获取原始数据
        item = self.dataset[idx]
        image = item['image']  # PIL.Image
        prompt = item['prompt']  # str
        
        # 2️⃣ 图像预处理
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # 处理 RGBA → RGB
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        
        pixel_values = self.image_transforms(image)
        # 结果形状: [3, 512, 512]，值域: [-1, 1]
        
        # 3️⃣ 文本预处理（Tokenization）
        text_inputs = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,  # 通常是 77
            padding="max_length",  # 填充到最大长度
            truncation=True,  # 如果超长则截断
            return_tensors="pt"  # 返回 PyTorch tensor
        )
        input_ids = text_inputs.input_ids  # 形状: [1, 77]
        
        # 4️⃣ 条件 Dropout（提高模型的无条件生成能力）
        # 以一定概率将 prompt 替换为空字符串对应的 token
        if torch.rand(1) < self.conditioning_dropout_prob:
            input_ids = self.tokenizer(
                "",  # 空提示
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                return_tensors="pt"
            ).input_ids
        
        # 5️⃣ 返回格式
        return {
            'pixel_values': pixel_values,       # [3, 512, 512] 范围 [-1, 1]
            'input_ids': input_ids.squeeze(0),  # [77] 方便 DataLoader
            'prompt': prompt                    # 字符串，用于日志或调试
        }


class DataLoaderFactory:
    """
    数据加载器工厂函数，方便创建训练/验证数据加载器
    """
    
    @staticmethod
    def create_train_dataloader(
        dataset,
        tokenizer,
        batch_size=4,
        num_workers=4,
        image_size=512,
        shuffle=True,
        pin_memory=True
    ):
        """创建训练数据加载器"""
        sd_dataset = StableDiffusionDataset(
            dataset=dataset,
            tokenizer=tokenizer,
            image_size=image_size,
            conditioning_dropout_prob=0.1
        )
        
        return DataLoader(
            sd_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True  # 训练时丢弃最后不完整的 batch
        )
    
    @staticmethod
    def create_val_dataloader(
        dataset,
        tokenizer,
        batch_size=4,
        num_workers=0,  # 验证通常单进程
        image_size=512,
    ):
        """创建验证数据加载器"""
        sd_dataset = StableDiffusionDataset(
            dataset=dataset,
            tokenizer=tokenizer,
            image_size=image_size,
            conditioning_dropout_prob=0.0  # 验证时不使用 dropout
        )
        
        return DataLoader(
            sd_dataset,
            batch_size=batch_size,
            shuffle=False,  # 验证不需要随机打乱
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False  # 验证要处理所有数据
        )

if __name__ == "__main__":
    import sys
    from datasets import load_from_disk
    
    print("=" * 60)
    print("🧪 Stable Diffusion Dataset Adapter 测试")
    print("=" * 60)
    
    try:
        # 1️⃣ 加载数据集
        print("\n[1/4] 加载数据集...")
        dataset_path = "data/lunara_final_dataset_rev"
        try:
            dataset = load_from_disk(dataset_path)
            print(f"✅ 数据集加载成功！")
            print(f"   - 数据集大小: {len(dataset)} 个样本")
            print(f"   - 字段: {dataset.column_names}")
        except FileNotFoundError:
            print(f"❌ 找不到数据集路径: {dataset_path}")
            print("   请确保已运行: python scripts/preprocessing/merge_labels.py")
            sys.exit(1)
        
        # 2️⃣ 加载 Tokenizer
        print("\n[2/4] 加载 CLIP Tokenizer...")
        try:
            tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
            print(f"✅ Tokenizer 加载成功！")
            print(f"   - Model Max Length: {tokenizer.model_max_length}")
            print(f"   - Vocab Size: {len(tokenizer)}")
        except Exception as e:
            print(f"❌ Tokenizer 加载失败: {e}")
            sys.exit(1)
        
        # 3️⃣ 创建 Dataset 实例
        print("\n[3/4] 创建 StableDiffusionDataset...")
        try:
            # 只用前 10 个样本测试
            test_dataset = dataset.select(range(min(10, len(dataset))))
            sd_dataset = StableDiffusionDataset(
                dataset=test_dataset,
                tokenizer=tokenizer,
                image_size=512,
                conditioning_dropout_prob=0.1
            )
            print(f"✅ Dataset 创建成功！")
            print(f"   - 数据集大小: {len(sd_dataset)}")
            
            # 测试单个样本
            print("\n   📄 测试单个样本...")
            sample = sd_dataset[0]
            print(f"   - pixel_values shape: {sample['pixel_values'].shape}")
            print(f"   - pixel_values range: [{sample['pixel_values'].min():.2f}, {sample['pixel_values'].max():.2f}]")
            print(f"   - input_ids shape: {sample['input_ids'].shape}")
            print(f"   - prompt: {sample['prompt'][:50]}...")
            
        except Exception as e:
            print(f"❌ Dataset 创建失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        # 4️⃣ 创建 DataLoader 并测试
        print("\n[4/4] 创建 DataLoader...")
        try:
            train_dataloader = DataLoaderFactory.create_train_dataloader(
                dataset=test_dataset,
                tokenizer=tokenizer,
                batch_size=2,
                num_workers=0,  # 测试时用 0 避免多进程问题
                image_size=512
            )
            print(f"✅ DataLoader 创建成功！")
            print(f"   - Batch size: 2")
            print(f"   - 总 batch 数: {len(train_dataloader)}")
            
            # 加载一个 batch 进行验证
            print("\n   📦 加载第一个 batch...")
            batch = next(iter(train_dataloader))
            
            print(f"   ✅ Batch 结构验证：")
            print(f"      - pixel_values: {batch['pixel_values'].shape}")
            print(f"        范围: [{batch['pixel_values'].min():.2f}, {batch['pixel_values'].max():.2f}]")
            print(f"      - input_ids: {batch['input_ids'].shape}")
            print(f"      - prompt: {len(batch['prompt'])} 个文本")
            
            # 验证数据类型
            assert batch['pixel_values'].dtype == torch.float32, "pixel_values 应该是 float32"
            assert batch['input_ids'].dtype == torch.long, "input_ids 应该是 long"
            assert batch['pixel_values'].shape == (2, 3, 512, 512), "pixel_values 形状不对"
            assert batch['input_ids'].shape == (2, 77), "input_ids 形状不对"
            
            print("\n   ✅ 形状和数据类型验证通过！")
            
        except Exception as e:
            print(f"❌ DataLoader 测试失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        # 验证数据集 - 检查不同的样本
        print("\n" + "=" * 60)
        print("✅ 所有测试通过！数据集已准备就绪。")
        print("=" * 60)
        
        print("\n📊 数据集摘要：")
        print(f"   - 总样本数: {len(dataset)}")
        print(f"   - 图像大小: 512 × 512")
        print(f"   - 文本编码长度: 77")
        print(f"   - 值域范围: [-1.0, 1.0]")
        print("\n✨ 已可开始 LoRA 微调训练！")
        
    except KeyboardInterrupt:
        print("\n⚠️  测试被中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
