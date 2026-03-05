import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve,
                             confusion_matrix, average_precision_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

FEATURE_COLS = [
    'sfari_score', 'cadd_score', 'conservation', 'gnomad_af',
    'brain_expression', 'domain_disrupted', 'is_lof',
    'proband_count', 'is_autism_gene',
    # Encoded variant effect
    'effect_nonsense', 'effect_frameshift', 'effect_splice_site',
    'effect_missense_damaging', 'effect_inframe_indel',
    'effect_missense_benign', 'effect_synonymous'
]

def encode_features(df):
    """One-hot encode variant effect"""
    effects = ['nonsense','frameshift','splice_site','missense_damaging',
               'inframe_indel','missense_benign','synonymous']
    for e in effects:
        df[f'effect_{e}'] = (df['variant_effect'] == e).astype(int)
    return df

def train():
    print("🧬 Loading autism de novo SNP dataset...")
    df = pd.read_csv("data/autism_snps.csv")
    df = encode_features(df)
    print(f"   {len(df):,} SNPs | "
          f"Pathogenic: {df['pathogenic'].sum():,} | "
          f"Benign: {(df['pathogenic']==0).sum():,}")

    X = df[FEATURE_COLS].values
    y = df['pathogenic'].values

    # Scale
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train models
    models = {
        'Random Forest':       RandomForestClassifier(
                                 n_estimators=300, max_depth=12,
                                 min_samples_leaf=2, random_state=42),
        'Gradient Boosting':   GradientBoostingClassifier(
                                 n_estimators=200, learning_rate=0.05,
                                 max_depth=5, random_state=42),
        'Logistic Regression': LogisticRegression(
                                 max_iter=1000, C=0.5, random_state=42),
    }

    best_auc, best_model, best_name = 0, None, ""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n🏋️  Training models...")
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        auc    = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
        cv_auc = cross_val_score(clf, X_scaled, y, cv=cv,
                                 scoring='roc_auc').mean()
        print(f"   {name:25s} ROC-AUC={auc:.4f}  CV={cv_auc:.4f}")
        if auc > best_auc:
            best_auc, best_model, best_name = auc, clf, name

    print(f"\n🏆 Best: {best_name} (ROC-AUC: {best_auc:.4f})")

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:,1]
    avg_pr = average_precision_score(y_test, y_prob)

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred,
          target_names=['Benign','Pathogenic']))

    # ── Plot: ROC Curve ──────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#E74C3C', lw=2,
            label=f'{best_name} (AUC = {best_auc:.4f})')
    ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.4, label='Random (AUC = 0.50)')
    ax.fill_between(fpr, tpr, alpha=0.08, color='#E74C3C')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve — Autism De Novo SNP Pathogenicity',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('models/plots/roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ roc_curve.png")

    # ── Plot: Confusion Matrix ───────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=['Benign','Pathogenic'],
                yticklabels=['Benign','Pathogenic'],
                ax=ax, linewidths=2, annot_kws={'size':14})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix\nAutism De Novo SNP Classifier',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('models/plots/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ confusion_matrix.png")

    # ── Plot: Feature Importance ─────────────────────────────
    importance = pd.DataFrame({
        'feature':    FEATURE_COLS,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=True).tail(12)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#E74C3C' if 'sfari' in f or 'cadd' in f or 'lof' in f or 'proband' in f
              else '#3498DB' for f in importance['feature']]
    ax.barh(importance['feature'], importance['importance'],
            color=colors, edgecolor='white')
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title('Top Features for Autism SNP Pathogenicity Prediction',
                 fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color='#E74C3C', label='Core clinical features'),
        Patch(color='#3498DB', label='Genomic features')
    ], fontsize=10)
    plt.tight_layout()
    plt.savefig('models/plots/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ feature_importance.png")

    # ── Plot: CADD score vs pathogenicity (scatter) ──────────
    fig, ax = plt.subplots(figsize=(9, 6))
    benign     = df[df['pathogenic']==0]
    pathogenic = df[df['pathogenic']==1]
    ax.scatter(benign['cadd_score'], benign['conservation'],
               alpha=0.3, s=15, color='#2ECC71', label='Benign')
    ax.scatter(pathogenic['cadd_score'], pathogenic['conservation'],
               alpha=0.3, s=15, color='#E74C3C', label='Pathogenic')
    ax.axvline(20, color='gray', linestyle='--', alpha=0.5, label='CADD > 20 (deleterious)')
    ax.set_xlabel('CADD Score', fontsize=12)
    ax.set_ylabel('Conservation Score (phyloP)', fontsize=12)
    ax.set_title('CADD Score vs Conservation\nPathogenic vs Benign De Novo SNPs',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('models/plots/cadd_conservation_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ cadd_conservation_scatter.png")

    # Save
    with open("models/classifier.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/feature_cols.pkl", "wb") as f:
        pickle.dump(FEATURE_COLS, f)

    pd.DataFrame([{
        "model":      best_name,
        "roc_auc":    round(best_auc, 4),
        "avg_pr":     round(avg_pr, 4),
        "n_train":    len(X_train),
        "n_test":     len(X_test),
    }]).to_csv("models/metrics.csv", index=False)

    print(f"\n✅ Done! ROC-AUC: {best_auc:.4f} | Avg Precision: {avg_pr:.4f}")

if __name__ == "__main__":
    train()
