"""
inspect_hdf5.py
扫描 ham_hdf5/ 目录下所有 HDF5 文件，列出每个文件中包含的:
  - 所有 dataset 名称
  - 数据类型 (dtype / Python type)
  - 数据形状 (shape) 或 大小
  - 数据内容预览 (前 200 字符)

运行: python inspect_hdf5.py
"""

import h5py
import os
from pathlib import Path

HDF5_DIR = Path("./ham_hdf5")

def inspect_item(name, obj, results):
    """递归回调: 收集 dataset 信息"""
    if isinstance(obj, h5py.Dataset):
        info = {
            "name": name,
            "shape": obj.shape,
            "dtype": str(obj.dtype),
            "size": obj.size,
        }
        # 尝试读取数据并展示预览
        try:
            data = obj[()]
            info["python_type"] = type(data).__name__
            if isinstance(data, bytes):
                preview = data.decode("utf-8", errors="replace")[:200]
                info["preview"] = preview
            elif hasattr(data, "shape"):
                info["preview"] = str(data.flat[:5] if data.size > 5 else data)
            else:
                info["preview"] = str(data)[:200]
        except Exception as e:
            info["preview"] = f"<读取失败: {e}>"
        results.append(info)
    elif isinstance(obj, h5py.Group):
        # 记录组信息
        results.append({
            "name": name,
            "type": "Group",
            "n_children": len(obj),
        })


def inspect_file(filepath):
    """检查单个 HDF5 文件，返回所有 dataset/group 信息"""
    results = []
    with h5py.File(filepath, "r") as f:
        f.visititems(lambda name, obj: inspect_item(name, obj, results))
    return results


def categorize_datasets(datasets):
    """按前缀对 dataset 名称分类 (如 ham_JW-, ham_BK-, ham_molec-)"""
    categories = {}
    for ds in datasets:
        if ds.get("type") == "Group":
            continue
        name = ds["name"]
        # 尝试提取前缀, 例如 "ham_JW-4" -> "ham_JW"
        parts = name.rsplit("-", 1)
        prefix = parts[0] if len(parts) == 2 and parts[1].isdigit() else name
        categories.setdefault(prefix, []).append(ds)
    return categories


def main():
    hdf5_files = sorted(HDF5_DIR.glob("*.hdf5"))
    if not hdf5_files:
        print(f"在 {HDF5_DIR} 中未找到 .hdf5 文件")
        return

    print(f"{'='*80}")
    print(f"  HDF5 文件检查报告")
    print(f"  目录: {HDF5_DIR.resolve()}")
    print(f"  共找到 {len(hdf5_files)} 个文件")
    print(f"{'='*80}\n")

    all_prefixes = set()

    for fpath in hdf5_files:
        size_mb = fpath.stat().st_size / (1024 * 1024)
        print(f"{'─'*80}")
        print(f"📄 文件: {fpath.name}  ({size_mb:.1f} MB)")
        print(f"{'─'*80}")

        datasets = inspect_file(fpath)

        # 分类统计
        categories = categorize_datasets(datasets)
        all_prefixes.update(categories.keys())

        # 按类别打印
        for prefix, items in sorted(categories.items()):
            print(f"\n  📂 类别: {prefix}  (共 {len(items)} 个 dataset)")
            for ds in items:
                print(f"    ├─ {ds['name']}")
                print(f"    │    dtype={ds['dtype']}, "
                      f"python_type={ds.get('python_type', '?')}, "
                      f"shape={ds['shape']}, size={ds['size']}")
                if "preview" in ds:
                    preview = ds["preview"].replace("\n", " ")[:120]
                    print(f"    │    预览: {preview}...")
                print()

    # 汇总
    print(f"\n{'='*80}")
    print(f"  汇总")
    print(f"{'='*80}")
    print(f"  所有文件中出现的 dataset 类别前缀:")
    for p in sorted(all_prefixes):
        print(f"    • {p}")
    print()


if __name__ == "__main__":
    main()
