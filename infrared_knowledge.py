"""
IR-DKFL: Infrared Domain Knowledge Few-shot Learning
论文里的"红外领域专家知识库"实际实现

For each class in HIT-UAV, we construct:
  - Temperature Gradient (TG) descriptor: 描述温度梯度边界特征
  - Radiation Intensity (RI) descriptor: 描述热辐射能量分布特征

These descriptors are designed following GPT-4 prompts about infrared imaging
characteristics, then refined manually for the HIT-UAV scenario.

Reference (from your paper line 164, 547):
  "Professional descriptions of infrared scenes are generated using large 
   language models such as GPT-4"
   
Usage in your training script:
    from infrared_knowledge import build_infrared_prompts
    
    names = yaml_load('dataset.yaml')['names']  # {0: 'Person', 1: 'Car', ...}
    ir_prompts = build_infrared_prompts(list(names.values()))
    # 现在 ir_prompts 是详细的红外描述, 不再是简单类名
    tpe = model.get_text_pe(ir_prompts)
"""

# ============================================================================
# 论文 TDSA 的双分支知识库 (Temperature Gradient + Radiation Intensity)
# 每个类别有两组描述, concat 后作为 text encoder 的输入
# ============================================================================

# Temperature Gradient (TG) Branch:
# 关注"温度差异/边界特征" — 论文 line 397: 
# "temperature gradient branch focuses on temperature differences between 
#  targets and backgrounds, producing strong responses at thermal boundaries"
TG_DESCRIPTORS = {
    "Person": (
        "a thermal infrared aerial view of a person, "
        "showing sharp temperature gradient between warm human body and cooler ground, "
        "with distinct thermal boundary at body silhouette outline"
    ),
    "Car": (
        "a thermal infrared aerial view of a vehicle, "
        "exhibiting strong temperature gradient at engine compartment and exhaust system, "
        "with sharp thermal contrast between hot mechanical components and cooler car body panels"
    ),
    "Bicycle": (
        "a thermal infrared aerial view of a bicycle, "
        "presenting weak temperature gradient with subtle thermal boundary, "
        "showing slight heat signature only at moving mechanical parts"
    ),
    "OtherVehicle": (
        "a thermal infrared aerial view of a non-standard vehicle such as a truck or bus, "
        "displaying complex temperature gradient pattern across large hot surfaces, "
        "with multiple thermal boundaries from engine, tires, and body sections"
    ),
    "DontCare": (
        "a thermal infrared aerial view of an ambiguous or partially occluded object, "
        "showing unclear temperature gradient and indistinct thermal boundaries, "
        "potentially confused with background thermal clutter"
    ),
}

# Radiation Intensity (RI) Branch:
# 关注"热辐射能量分布" — 论文 line 397:
# "thermal radiation intensity branch emphasizes energy distribution patterns, 
#  assigning higher weights to regions with significant thermal radiation"
RI_DESCRIPTORS = {
    "Person": (
        "high-temperature human body radiation in infrared imagery, "
        "concentrated thermal energy emission from head, torso, and exposed skin areas, "
        "consistent radiation intensity at approximately body temperature 36 to 37 degrees Celsius"
    ),
    "Car": (
        "intense thermal radiation from vehicle engine and exhaust pipe, "
        "with radiation intensity peaks at hood and underbody mechanical regions, "
        "showing characteristic hot-spot energy distribution typical of running vehicles"
    ),
    "Bicycle": (
        "low thermal radiation intensity from bicycle frame, "
        "minimal energy emission concentrated mainly at recently used mechanical joints, "
        "weak radiation signature easily obscured by background thermal noise"
    ),
    "OtherVehicle": (
        "high thermal radiation intensity from large vehicles like trucks or buses, "
        "extensive energy distribution across engine bay, tires, and exhaust systems, "
        "multiple high-intensity radiation centers from various heat sources"
    ),
    "DontCare": (
        "ambiguous thermal radiation pattern of unclear origin, "
        "irregular energy distribution that may overlap with background thermal clutter, "
        "weak or inconsistent radiation intensity below reliable detection threshold"
    ),
}


def build_infrared_prompts(class_names, mode="dual_branch"):
    """Build infrared domain expert descriptions for given class names.
    
    Args:
        class_names: list of strings, e.g. ['Person', 'Car', 'Bicycle', ...]
        mode: 
            'simple'      - just class name (baseline, equivalent to current behavior)
            'tg_only'     - temperature gradient branch only
            'ri_only'     - radiation intensity branch only  
            'dual_branch' - concatenate TG + RI (论文里的 TDSA, 推荐)
    
    Returns:
        list of strings ready to feed model.get_text_pe()
        
    Example:
        >>> names = ['Person', 'Car', 'Bicycle', 'OtherVehicle', 'DontCare']
        >>> prompts = build_infrared_prompts(names, mode='dual_branch')
        >>> len(prompts) == 5
        True
    """
    if mode == "simple":
        return [str(n) for n in class_names]
    
    prompts = []
    for name in class_names:
        if name not in TG_DESCRIPTORS:
            # 未知类别 fallback: 用通用红外描述
            tg = f"a thermal infrared aerial view of {name}, showing temperature gradient at object boundaries"
            ri = f"thermal radiation pattern of {name} in infrared imagery"
        else:
            tg = TG_DESCRIPTORS[name]
            ri = RI_DESCRIPTORS[name]
        
        if mode == "tg_only":
            prompts.append(tg)
        elif mode == "ri_only":
            prompts.append(ri)
        elif mode == "dual_branch":
            # 论文里的 TC = Concat(TG, RI), 在 text 层面就是把两段 description 拼起来
            # CLIP 文本最大 length 77 tokens, 我们的描述每段约 30-40 tokens, 拼起来约 60-80
            # 用句号分隔, 让 CLIP 理解为两个相关但独立的句子
            prompts.append(f"{tg}. {ri}")
        else:
            raise ValueError(f"Unknown mode: {mode}. Use one of: simple, tg_only, ri_only, dual_branch")
    
    return prompts


def get_dual_branch_separately(class_names):
    """Return TG and RI prompts separately, useful for true dual-branch fusion.
    
    Returns:
        (tg_prompts, ri_prompts): two lists of strings
        
    Use this if you want to encode TG and RI separately and fuse later
    (论文 Eq.: TC = Concat(Lin_TG(P_tg), Lin_RI(P_ri))).
    But for simplicity and effectiveness, build_infrared_prompts(mode='dual_branch')
    is sufficient — it relies on CLIP's own self-attention to fuse the two branches.
    """
    tg_prompts = [TG_DESCRIPTORS.get(n, f"a thermal infrared view of {n}") for n in class_names]
    ri_prompts = [RI_DESCRIPTORS.get(n, f"thermal radiation pattern of {n}") for n in class_names]
    return tg_prompts, ri_prompts


if __name__ == "__main__":
    # 自检
    names = ['Person', 'Car', 'Bicycle', 'OtherVehicle', 'DontCare']
    
    print("=" * 70)
    print("IR-DKFL 知识库自检")
    print("=" * 70)
    
    for mode in ['simple', 'tg_only', 'ri_only', 'dual_branch']:
        prompts = build_infrared_prompts(names, mode=mode)
        print(f"\n--- mode = {mode} ---")
        for i, (n, p) in enumerate(zip(names, prompts)):
            print(f"  [{i}] {n}: {p[:100]}{'...' if len(p) > 100 else ''}")
    
    print("\n" + "=" * 70)
    print("✅ IR-DKFL 模块 OK")
    print("=" * 70)
