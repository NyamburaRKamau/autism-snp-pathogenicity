import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("data", exist_ok=True)
os.makedirs("models/plots", exist_ok=True)

np.random.seed(42)

# Real autism-associated genes with SFARI scores (1=highest confidence)
# Source: SFARI Gene database sfari.org/resource/sfari-gene
SFARI_GENES = {
    # Score 1 — High confidence autism genes
    "ADNP":   1, "ANK2":   1, "ANKRD11": 1, "ARID1B": 1,
    "ASXL3":  1, "CHD2":   1, "CHD8":    1, "CNTNAP2":1,
    "CTNNB1": 1, "DYRK1A": 1, "FOXP1":   1, "GRIN2B": 1,
    "KDM5C":  1, "KDM6A":  1, "MED13L":  1, "POGZ":   1,
    "PTEN":   1, "SCN1A":  1, "SCN2A":   1, "SETD5":  1,
    "SHANK2": 1, "SHANK3": 1, "SYNGAP1": 1, "TBR1":   1,
    # Score 2 — Strong autism genes
    "ACHE":   2, "ADSL":   2, "AKAP9":   2, "CNTN4":  2,
    "CNTN6":  2, "DPP10":  2, "EN2":     2, "FOXP2":  2,
    "GRIN2A": 2, "NLGN3":  2, "NLGN4X":  2, "NRXN1":  2,
    "NRXN3":  2, "SHANK1": 2, "SLC6A4":  2, "TSC1":   2,
    # Score 3 — Suggestive autism genes
    "ABCA7":  3, "ASTN2":  3, "AUTS2":   3, "CACNA1C":3,
    "CACNA1H":3, "CNTN5":  3, "DISC1":   3, "FMR1":   3,
    "GABRB3": 3, "LAMB1":  3, "MBD5":    3, "PARK2":  3,
}

# Control genes (not autism-associated)
CONTROL_GENES = [
    "BRCA1","BRCA2","TP53","EGFR","KRAS","BRAF","ALK",
    "MYC","RB1","VHL","APC","MLH1","MSH2","ATM","PALB2",
    "RAD51","CHEK2","CDH1","STK11","SMAD4","TTN","MUC16",
    "OBSCN","AHNAK2","SYNE1","PCLO","FLG","PLEC","DNAH5"
]

VARIANT_EFFECTS = {
    "nonsense":         0.90,  # stop gained — high pathogenicity
    "frameshift":       0.88,  # frameshift — high pathogenicity
    "splice_site":      0.82,  # splice disruption
    "missense_damaging":0.72,  # damaging missense
    "inframe_indel":    0.55,  # inframe insertion/deletion
    "missense_benign":  0.25,  # benign missense
    "synonymous":       0.08,  # synonymous — usually benign
}

def generate_snp(gene, sfari_score, is_autism_gene):
    """Generate a realistic de novo SNP with biological features"""
    
    # Variant effect — autism-associated genes more likely to have damaging variants
    if is_autism_gene:
        effect_weights = [0.15, 0.15, 0.15, 0.30, 0.10, 0.10, 0.05]
    else:
        effect_weights = [0.05, 0.05, 0.08, 0.20, 0.10, 0.30, 0.22]
    
    effects = list(VARIANT_EFFECTS.keys())
    effect  = np.random.choice(effects, p=effect_weights)
    base_path = VARIANT_EFFECTS[effect]
    
    # Conservation score (phyloP -20 to +20, higher = more conserved = more pathogenic)
    if is_autism_gene:
        conservation = np.random.normal(8.5, 3.0)
    else:
        conservation = np.random.normal(2.0, 4.0)
    conservation = np.clip(conservation, -10, 20)
    
    # CADD score (0-60, >20 = likely deleterious, >30 = highly deleterious)
    if is_autism_gene and effect in ["nonsense","frameshift","splice_site"]:
        cadd = np.random.normal(38, 6)
    elif is_autism_gene:
        cadd = np.random.normal(24, 8)
    else:
        cadd = np.random.normal(12, 8)
    cadd = np.clip(cadd, 0, 60)
    
    # gnomAD allele frequency (de novo = very rare, <0.0001)
    if is_autism_gene:
        gnomad_af = np.random.exponential(0.00005)
    else:
        gnomad_af = np.random.exponential(0.0002)
    gnomad_af = np.clip(gnomad_af, 0, 0.01)
    
    # Brain expression (TPM, autism genes highly expressed in brain)
    if is_autism_gene:
        brain_expr = np.random.lognormal(3.5, 0.8)
    else:
        brain_expr = np.random.lognormal(2.0, 1.0)
    
    # Protein domain disruption
    domain_disrupted = int(
        is_autism_gene and
        effect in ["nonsense","frameshift","missense_damaging"] and
        np.random.random() > 0.35
    )
    
    # Loss of function
    is_lof = int(effect in ["nonsense","frameshift","splice_site"])
    
    # SFARI score (0 = not in SFARI)
    sfari = sfari_score if is_autism_gene else 0
    
    # Proband count (how many autism probands carry this variant)
    if is_autism_gene:
        proband_count = np.random.poisson(3.5) + 1
    else:
        proband_count = np.random.poisson(0.3)
    
    # Pathogenicity label — probabilistic based on features
    path_score = (
        (base_path * 0.30) +
        (min(cadd / 60, 1.0) * 0.25) +
        (min(conservation / 20, 1.0) * 0.15) +
        ((sfari_score == 1) * 0.15 if is_autism_gene else 0) +
        (domain_disrupted * 0.10) +
        (is_lof * 0.05)
    )
    
    # Add noise
    path_score += np.random.normal(0, 0.08)
    label = int(path_score > 0.50)
    
    return {
        "gene":              gene,
        "sfari_score":       sfari,
        "variant_effect":    effect,
        "cadd_score":        round(cadd, 2),
        "conservation":      round(conservation, 3),
        "gnomad_af":         round(gnomad_af, 6),
        "brain_expression":  round(brain_expr, 2),
        "domain_disrupted":  domain_disrupted,
        "is_lof":            is_lof,
        "proband_count":     proband_count,
        "is_autism_gene":    int(is_autism_gene),
        "pathogenic":        label
    }


def generate_dataset(n=3000):
    print("🧬 Generating autism de novo SNP dataset...")
    rows = []

    # 60% from autism-associated genes, 40% from control genes
    n_autism   = int(n * 0.60)
    n_control  = n - n_autism

    for _ in range(n_autism):
        gene       = np.random.choice(list(SFARI_GENES.keys()))
        sfari_score = SFARI_GENES[gene]
        rows.append(generate_snp(gene, sfari_score, is_autism_gene=True))

    for _ in range(n_control):
        gene = np.random.choice(CONTROL_GENES)
        rows.append(generate_snp(gene, 0, is_autism_gene=False))

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"   Total SNPs      : {len(df):,}")
    print(f"   Pathogenic      : {df['pathogenic'].sum():,}")
    print(f"   Benign          : {(df['pathogenic']==0).sum():,}")
    print(f"   Autism genes    : {df['is_autism_gene'].sum():,}")
    print(f"   SFARI Score 1   : {(df['sfari_score']==1).sum():,}")

    # ── EDA Plots ────────────────────────────────────────────────
    print("\n📊 Generating EDA plots...")

    # Plot 1: CADD score distribution by pathogenicity
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (col, label) in zip(axes, [
        ("cadd_score",     "CADD Score"),
        ("conservation",   "Conservation (phyloP)"),
        ("brain_expression","Brain Expression (TPM)")
    ]):
        for label_val, color, name in [(1,"#E74C3C","Pathogenic"),(0,"#2ECC71","Benign")]:
            subset = df[df['pathogenic']==label_val][col]
            ax.hist(subset, bins=35, alpha=0.6, color=color, label=name, edgecolor='white')
        ax.set_title(label, fontweight='bold', fontsize=12)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle("Feature Distributions: Pathogenic vs Benign De Novo SNPs",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("models/plots/feature_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ feature_distributions.png")

    # Plot 2: Variant effect vs pathogenicity
    fig, ax = plt.subplots(figsize=(11, 5))
    effect_path = df.groupby("variant_effect")["pathogenic"].mean().sort_values(ascending=False)
    colors = ["#E74C3C" if v > 0.5 else "#2ECC71" for v in effect_path.values]
    bars = ax.bar(effect_path.index, effect_path.values, color=colors, edgecolor='white')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Decision boundary')
    ax.set_title("Pathogenicity Rate by Variant Effect", fontweight='bold', fontsize=13)
    ax.set_xlabel("Variant Effect")
    ax.set_ylabel("Proportion Pathogenic")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, effect_path.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f"{val:.2f}", ha='center', fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("models/plots/variant_effect_pathogenicity.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ variant_effect_pathogenicity.png")

    # Plot 3: SFARI score vs pathogenicity
    fig, ax = plt.subplots(figsize=(8, 5))
    sfari_path = df.groupby("sfari_score")["pathogenic"].mean()
    sfari_labels = {0:"Not in SFARI", 1:"Score 1\n(High confidence)", 2:"Score 2\n(Strong)", 3:"Score 3\n(Suggestive)"}
    colors = ["#95A5A6","#E74C3C","#E67E22","#F1C40F"]
    ax.bar([sfari_labels.get(k,k) for k in sfari_path.index],
           sfari_path.values, color=colors[:len(sfari_path)], edgecolor='white')
    ax.set_title("Pathogenicity Rate by SFARI Gene Score", fontweight='bold', fontsize=13)
    ax.set_ylabel("Proportion Pathogenic")
    ax.set_ylim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("models/plots/sfari_pathogenicity.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ sfari_pathogenicity.png")

    df.to_csv("data/autism_snps.csv", index=False)
    print(f"\n💾 Saved to data/autism_snps.csv")
    return df

if __name__ == "__main__":
    generate_dataset()
