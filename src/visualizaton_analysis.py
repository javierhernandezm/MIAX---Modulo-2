import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def pie_count_volume_chart(df, group_col, title_prefix):
    dist_group_col = df[group_col].value_counts()
    print(f"Distribución de {group_col}:")
    print(dist_group_col)
    print(f"\nTotal de {group_col} diferentes: {len(dist_group_col)}")

    df_grouped = df.groupby(group_col).agg({
        'ISIN': 'count', 
        'Outstanding Amount': 'sum'
    }).rename(columns={'ISIN': 'Count', 'Outstanding Amount': 'Volume'})

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'{title_prefix}: Distribución por Número vs Volumen', fontsize=16)

    df_grouped['Count'].plot.pie(
        ax=axes[0], autopct='%1.1f%%', startangle=90
    )
    axes[0].set_title('% Número de Bonos')

    df_grouped['Volume'].plot.pie(
        ax=axes[1], autopct='%1.1f%%', startangle=90
    )
    axes[1].set_title('% Volumen (Outstanding Amount)')

    plt.tight_layout()
    plt.show()

def divisas_visualization(df):
    pie_count_volume_chart(df, 'Ccy', 'Divisas')

def callable_visualization(df):
    pie_count_volume_chart(df, 'Callable', 'Callable vs Non-Callable')

def tipo_cupon_visualization(df):
    pie_count_volume_chart(df, 'Coupon Type', 'Tipo de Cupón')

def frequencia_cupon_visualization(df):
    pie_count_volume_chart(df, 'Coupon Frequency', 'Frecuencia de Cupón')   

def senioridade_visualization(df):
    print("\Senioridade:")
    print(df['Seniority'].value_counts())
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(data=df, x='Seniority', order=df['Seniority'].value_counts().index, ax=ax)
    ax.set_title('Distribución de Senioridade')
    ax.set_xlabel('Senioridade')
    ax.set_ylabel('Número de Bonos')
    plt.show()


def distribuicion_por_setores_visualization(df):
    print("Distribuição por setor:")
    setores = df['Industry Sector'].value_counts()
    setores.sort_values(ascending=True, inplace=True)
    print(setores)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Setores: Distribución por Número de Bonos', fontsize=16)

    axes[0].pie(setores.values, labels=setores.index, autopct='%1.1f%%', startangle=90)
    axes[0].set_title('Distribución de Setores')

    axes[1].barh(y=setores.index, width=setores.values)
    axes[1].set_title('Distribución por Setor (Todos)')
    axes[1].set_xlabel('Número de Bonos')
    axes[1].set_ylabel('Setor')
    axes[1].tick_params(axis='y', labelsize=8)

    print(f"\nAnálisis de diversificación:")
    print(f"Total de sectores distintos: {len(setores)}")
    print(f"Razón bonos/sectores: {len(df)/len(setores):.2f}")
    print(f"Concentración:")
    print(f"- Top 3 sectores representan: {(setores.head(3).sum()/len(df)*100):.1f}% dos bonos")
    print(f"- Top 5 sectores representan: {(setores.head(5).sum()/len(df)*100):.1f}% dos bonos")
    print(f"- Top 10 sectores representan: {(setores.head(10).sum()/len(df)*100):.1f}% dos bonos")
    

    plt.tight_layout()
    plt.show()

def distribuicion_por_emissores_visualization(df):
    print("\nDistribución por emisor:")
    emissores = df['Issuer'].value_counts()
    print(f"Total de emissores: {len(emissores)}")

    total_emissores = len(emissores)
    cutoff_80_emissores = int(total_emissores * 0.8)
    emissores_top80 = emissores.iloc[:cutoff_80_emissores]
    emissores_outros = emissores.iloc[cutoff_80_emissores:].sum()

    emissores_pizza = emissores_top80.copy()
    if emissores_outros > 0:
        emissores_pizza['Outros'] = emissores_outros

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Emissores: Distribución por Número de Bonos', fontsize=16)

    emissores_pizza_viz = emissores.head(10).copy()
    emissores_outros_viz = emissores.iloc[10:].sum()
    if emissores_outros_viz > 0:
        emissores_pizza_viz['Outros'] = emissores_outros_viz

    axes[0].pie(emissores_pizza_viz.values, labels=emissores_pizza_viz.index, autopct='%1.1f%%', 
            startangle=90, textprops={'fontsize': 8})
    axes[0].set_title('Distribución de Emissores (Top 10 + Outros)')

    top_20_emissores = emissores.head(20)
    print(f"\nTop 20 emissores:")
    print(top_20_emissores)

    top_20_emissores.sort_values(ascending=True, inplace=True)
    axes[1].barh(y=top_20_emissores.index, width=top_20_emissores.values)
    axes[1].set_title('Top 20 Emissores por Número de Bonos')
    axes[1].set_xlabel('Número de Bonos')
    axes[1].set_ylabel('Emissor')
    axes[1].tick_params(axis='y', labelsize=8)

    plt.tight_layout()
    plt.show()

    print(f"\nAnálisis de diversificación:")
    print(f"Total de emissores distintos: {len(emissores)}")
    print(f"Razón bonos/emissores: {len(df)/len(emissores):.2f}")
    print(f"Concentración:")
    print(f"- Top 3 emissores representan: {(emissores.head(5).sum()/len(df)*100):.1f}% dos bonos")
    print(f"- Top 5 emissores representan: {(emissores.head(5).sum()/len(df)*100):.1f}% dos bonos")
    print(f"- Top 10 emissores representan: {(emissores.head(10).sum()/len(df)*100):.1f}% dos bonos")


def clasificar_rating(rating):
    ig_ratings = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-']
    return 'Investment Grade' if rating in ig_ratings else 'High Yield'

def distribuicion_ratings_visualization(df):
    print("Distribución de ratings:")
    ratings = df['Rating'].value_counts().sort_index()
    print(ratings)
    ratings = ratings.sort_values(ascending=False)

    df['Rating_Type'] = df['Rating'].apply(clasificar_rating)
    print(f"\nDistribución IG vs HY:")
    print(df['Rating_Type'].value_counts())

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Análisis de Ratings Crediticios', fontsize=16)

    ratings_sorted = ratings.sort_values(ascending=True)
    axes[0].barh(y=ratings_sorted.index, width=ratings_sorted.values)
    axes[0].set_title('Distribución por Rating Individual')
    axes[0].set_xlabel('Número de Bonos')
    axes[0].set_ylabel('Rating')
    axes[0].tick_params(axis='y', labelsize=10)

    for i, (rating, count) in enumerate(ratings_sorted.items()):
        axes[0].text(count + 5, i, str(count), va='center', fontsize=9)

    rating_type_counts = df['Rating_Type'].value_counts()
    colors = ['lightblue', 'orange']
    axes[1].pie(rating_type_counts.values, labels=rating_type_counts.index, 
            autopct='%1.1f%%', startangle=90, colors=colors)
    axes[1].set_title('Investment Grade vs High Yield')

    plt.tight_layout()
    plt.show()

    print(f"\nAnálisis de riesgo crediticio:")
    print(f"Total Investment Grade: {rating_type_counts['Investment Grade']} bonos ({rating_type_counts['Investment Grade']/len(df)*100:.1f}%)")
    print(f"Total High Yield: {rating_type_counts['High Yield']} bonos ({rating_type_counts['High Yield']/len(df)*100:.1f}%)")
    print(f"\nRatings más comunes:")
    print(f"- Rating más frecuente: {ratings.index[0]} ({ratings.iloc[0]} bonos)")
    print(f"- Top 3 ratings: {', '.join(ratings.head(3).index.tolist())}")

    rating_volume = df.groupby('Rating')['Outstanding Amount'].sum().sort_values(ascending=False)
    print(f"\nVolumen por rating (Top 5):")
    for rating, volume in rating_volume.head(5).items():
        percentage = volume / df['Outstanding Amount'].sum() * 100
        print(f"- {rating}: {volume/1e9:.1f}B EUR ({percentage:.1f}%)")

def tamanos_emision_visualization(df):
    plt.figure(figsize=(12, 6))

    sns.histplot(
        df['Outstanding Amount'], 
        bins=50, 
        kde=True, 
        edgecolor='black'
    )

    plt.title('Distribución de Tamaños de Emisión (Outstanding Amount)', fontsize=16)
    plt.xlabel('Volumen de la Emisión (en moneda original)')
    plt.ylabel('Frecuencia (Número de Bonos)')

    plt.axvline(df['Outstanding Amount'].mean(), color='red', linestyle='--', label=f"Media: {df['Outstanding Amount'].mean():,.0f}")
    plt.axvline(df['Outstanding Amount'].median(), color='orange', linestyle='-', label=f"Mediana: {df['Outstanding Amount'].median():,.0f}")

    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plt.show()

def seneority_visualization(df):
    orden_seniority = [
        '1st lien', 'Secured', 'Sr Preferred', 'Sr Unsecured', 
        'Sr Non Preferred', 'Subordinated', 'Jr Subordinated'
    ]

    df_sen = df.groupby('Seniority').agg({
        'Description': 'count', 
        'Outstanding Amount': 'sum'
    }).reindex(orden_seniority).dropna()

    df_sen_pct = df_sen.apply(lambda x: x / x.sum() * 100)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle('Estructura de Capital: Conteo vs Volumen', fontsize=16)

    sns.barplot(x=df_sen_pct.index, y=df_sen_pct['Description'], ax=axes[0], palette='Blues_r')
    axes[0].set_title('A) Distribución por Número de Bonos')
    axes[0].set_ylabel('% del Total de Bonos')
    axes[0].tick_params(axis='x', rotation=45)

    for i, v in enumerate(df_sen_pct['Description']):
        axes[0].text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=10)

    sns.barplot(x=df_sen_pct.index, y=df_sen_pct['Outstanding Amount'], ax=axes[1], palette='Blues_r')
    axes[1].set_title('B) Distribución por Volumen (Dinero invertido)')
    axes[1].set_ylabel('% del Total de Dinero')
    axes[1].tick_params(axis='x', rotation=45)

    for i, v in enumerate(df_sen_pct['Outstanding Amount']):
        axes[1].text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

def ratings_dist_visualization(df):
    orden_ratings = [
        'AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 
        'BBB+', 'BBB', 'BBB-', 'BB+', 'NR', 'Sin Rating'
    ]

    df_temp = df.copy()
    df_temp['Rating'] = df_temp['Rating'].fillna('Sin Rating')

    df_rat = df_temp.groupby('Rating').agg({
        'Description': 'count', 
        'Outstanding Amount': 'sum'
    }).reindex(orden_ratings).dropna()

    df_rat_pct = df_rat.apply(lambda x: x / x.sum() * 100)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle('Riesgo de Crédito: Conteo vs Volumen', fontsize=16)

    sns.barplot(x=df_rat_pct.index, y=df_rat_pct['Description'], ax=axes[0], palette='RdYlGn_r')
    axes[0].set_title('A) Distribución por Número de Bonos')
    axes[0].set_ylabel('% del Total de Bonos')
    axes[0].tick_params(axis='x', rotation=45)

    for i, v in enumerate(df_rat_pct['Description']):
        axes[0].text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)

    sns.barplot(x=df_rat_pct.index, y=df_rat_pct['Outstanding Amount'], ax=axes[1], palette='RdYlGn_r')
    axes[1].set_title('B) Distribución por Volumen (Dinero invertido)')
    axes[1].set_ylabel('% del Total de Dinero')
    axes[1].tick_params(axis='x', rotation=45)

    for i, v in enumerate(df_rat_pct['Outstanding Amount']):
        axes[1].text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()


def riesgo_liquidez(df):
    df['Bid_Ask_Spread_Pct'] = ((df['Ask Price'] - df['Bid Price']) / df['Price']) * 100

    print("=== ANÁLISIS DE RIESGO DE LIQUIDEZ ===")
    print("\nEstadísticas del spread bid-ask (%):")
    print(df['Bid_Ask_Spread_Pct'].describe())

    print("\n=== ANÁLISIS DEL NOMINAL VIVO ===")
    print("\nEstadísticas del nominal vivo (EUR):")
    print(df['Outstanding Amount'].describe())

    print(f"\nNominal vivo en billones EUR:")
    print(f"- Total: {df['Outstanding Amount'].sum()/1e9:.1f}B EUR")
    print(f"- Promedio: {df['Outstanding Amount'].mean()/1e6:.1f}M EUR")
    print(f"- Mediana: {df['Outstanding Amount'].median()/1e6:.1f}M EUR")

    print(f"\nDistribución por tamaño de emisión:")
    ranges = [0, 500e6, 1e9, 2e9, 5e9, float('inf')]
    labels = ['<500M', '500M-1B', '1B-2B', '2B-5B', '>5B']
    df['Size_Range'] = pd.cut(df['Outstanding Amount'], bins=ranges, labels=labels, include_lowest=True)
    size_dist = df['Size_Range'].value_counts().sort_index()
    for size_range, count in size_dist.items():
        percentage = count / len(df) * 100
        print(f"- {size_range}: {count} bonos ({percentage:.1f}%)")

    print(f"\n=== ANÁLISIS DE CUPONES ===")
    print("\nEstadísticas de cupones (%):")
    print(df['Coupon'].describe())

    print(f"\n=== ANÁLISIS DE PROBABILIDAD DE DEFAULT ===")
    print("\nEstadísticas de PD 1 año (%):")
    print(df['PD 1YR'].describe())

    plt.figure(figsize=(12, 6))

    sns.histplot(
        df['Bid_Ask_Spread_Pct'], 
        bins=50, 
        kde=True, 
        edgecolor='black',
        alpha=0.6
    )

    plt.title('Distribución del Riesgo de Liquidez (Bid-Ask Spread %)', fontsize=16)
    plt.xlabel('Spread (%) - Coste de Liquidez')
    plt.ylabel('Frecuencia (Número de Bonos)')

    mean_spread = df['Bid_Ask_Spread_Pct'].mean()
    median_spread = df['Bid_Ask_Spread_Pct'].median()

    plt.axvline(mean_spread, color='red', linestyle='--', linewidth=2, label=f"Media: {mean_spread:.3f}%")
    plt.axvline(median_spread, color='blue', linestyle='-.', linewidth=2, label=f"Mediana: {median_spread:.3f}%")

    plt.axvspan(0, 0.5, color='green', alpha=0.1, label='Alta Liquidez (<0.5%)')
    plt.axvspan(0.5, 1.5, color='yellow', alpha=0.1, label='Liquidez Media')

    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plt.show()

    print(f"Spread Promedio: {mean_spread:.3f}%")
    print(f"Spread Máximo: {df['Bid_Ask_Spread_Pct'].max():.3f}% (Posible iliquidez extrema)")

def distribuition_liquidez_visualization(df):
    if 'Bid_Ask_Spread_Pct' not in df.columns:
        df['Bid_Ask_Spread_Pct'] = ((df['Ask Price'] - df['Bid Price']) / df['Price']) * 100

    bins = [0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, float('inf')]

    labels = [
        '0-25 bps', '25-50 bps', '50-75 bps', '75-100 bps', 
        '100-125 bps', '125-150 bps', '150-175 bps', '175-200 bps', '> 200 bps'
    ]

    df['Liquidity_Bucket'] = pd.cut(df['Bid_Ask_Spread_Pct'], bins=bins, labels=labels, right=False)

    liquidity_dist = df['Liquidity_Bucket'].value_counts(normalize=True).sort_index() * 100

    plt.figure(figsize=(14, 7))

    ax = sns.barplot(
        x=liquidity_dist.index, 
        y=liquidity_dist.values
    )

    plt.title('Distribución de Liquidez por Tramos de 25 bps', fontsize=16)
    plt.xlabel('Rango de Bid-Ask Spread')
    plt.ylabel('Porcentaje de Bonos (%)')
    plt.xticks(rotation=45)

    for i, v in enumerate(liquidity_dist.values):
        ax.text(i, v + 0.2, f'{v:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    acumulado_50bps = liquidity_dist['0-25 bps'] + liquidity_dist['25-50 bps']
    plt.text(1, liquidity_dist.max()/2, f'Bonos muy líquidos (<50bps):\n{acumulado_50bps:.1f}%', 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='green'), fontsize=12, color='green')

    plt.tight_layout()
    plt.show()