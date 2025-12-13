from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum, avg, when, round
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

spark = SparkSession.builder \
    .appName("SQLDataAnalysis") \
    .master("local[*]") \
    .getOrCreate()
    
df = spark.read.option("header", True).option("inferSchema", True).option("sep", ";").csv("data.csv")
df = df.filter(df.ID != "Код")
 
df.createOrReplaceTempView("food_facilities")

print("=" * 80)
print("АНАЛИЗ ДАННЫХ О ЗАВЕДЕНИЯХ ОБЩЕСТВЕННОГО ПИТАНИЯ МОСКВЫ")
print("=" * 80)


print("\n" + "="*80)
print("УПРАЖНЕНИЕ 1: ОПИСАНИЕ НАБОРА ДАННЫХ")
print("="*80)

print(f"Общее количество записей: {df.count():,}")
print(f"Количество столбцов: {len(df.columns)}")
print("\nСхема данных:")
df.printSchema()

print("\n" + "-"*80)
print("ДЕТАЛЬНАЯ ИНФОРМАЦИЯ О СТОЛБЦАХ:")
print("-"*80)

column_info = []
for column in df.columns:
    null_count = df.filter(col(column).isNull()).count()
    unique_count = df.select(column).distinct().count()
    data_type = dict(df.dtypes)[column]
    column_info.append({
        'Столбец': column,
        'Тип данных': data_type,
        'Пустых значений': null_count,
        'Уникальных значений': unique_count
    })

info_df = pd.DataFrame(column_info)
print(info_df.to_string(index=False))

print("\n" + "-"*80)
print("ПРИМЕР ДАННЫХ (первые 5 строк):")
print("-"*80)
df.show(5, truncate=False)
 
 

print("\n" + "="*80)
print("УПРАЖНЕНИЕ 2: РАСПРЕДЕЛЕНИЕ ЗАВЕДЕНИЙ ПО АДМИНИСТРАТИВНЫМ ОКРУГАМ")
print("="*80)

district_distribution = spark.sql("""
SELECT 
    AdmArea as adm_area,
    COUNT(*) as count_establishments,
    ROUND(AVG(CAST(SeatsCount AS INT)), 1) as avg_seats
FROM food_facilities
WHERE AdmArea IS NOT NULL
GROUP BY AdmArea
ORDER BY count_establishments DESC
""")

print("Распределение заведений по административным округам:")

district_distribution = district_distribution.withColumnRenamed("adm_area", "Административный_округ") \
    .withColumnRenamed("count_establishments", "Количество_заведений") \
    .withColumnRenamed("avg_seats", "Среднее_посадочных_мест")
district_distribution.show(truncate=False)

district_pdf = district_distribution.toPandas()

plt.figure(figsize=(12, 6))
ax = plt.subplot(1, 2, 1)
bars = ax.barh(district_pdf['Административный_округ'], 
                district_pdf['Количество_заведений'], 
                color='steelblue')
ax.set_xlabel('Количество заведений')
ax.set_title('Количество заведений по округам')
ax.invert_yaxis()

for bar in bars:
    width = bar.get_width()
    ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
            f'{int(width)}', ha='left', va='center')

ax2 = plt.subplot(1, 2, 2)
ax2.barh(district_pdf['Административный_округ'], 
         district_pdf['Среднее_посадочных_мест'], 
         color='darkorange')
ax2.set_xlabel('Среднее посадочных мест')
ax2.set_title('Средняя вместимость по округам')
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig('распределение_по_округам.png', dpi=300, bbox_inches='tight')
plt.show()



print("\n" + "="*80)
print("УПРАЖНЕНИЕ 3: АНАЛИЗ ПО ТИПАМ ЗАВЕДЕНИЙ")
print("="*80)

type_analysis = spark.sql("""
SELECT 
    TypeObject as type_object,
    COUNT(*) as count,
    SUM(CAST(SeatsCount AS INT)) as total_seats,
    ROUND(AVG(CAST(SeatsCount AS INT)), 1) as avg_seats,
    MIN(CAST(SeatsCount AS INT)) as min_seats,
    MAX(CAST(SeatsCount AS INT)) as max_seats
FROM food_facilities
WHERE TypeObject IS NOT NULL
GROUP BY TypeObject
HAVING COUNT(*) >= 10
ORDER BY count DESC
""")

print("Статистика по типам заведений:")
type_analysis = type_analysis.withColumnRenamed("type_object", "Тип_заведения") \
    .withColumnRenamed("count", "Количество") \
    .withColumnRenamed("total_seats", "Всего_посадочных_мест") \
    .withColumnRenamed("avg_seats", "Среднее_посадочных_мест") \
    .withColumnRenamed("min_seats", "Минимальное_мест") \
    .withColumnRenamed("max_seats", "Максимальное_мест")
type_analysis.show(truncate=False)

type_pdf = type_analysis.toPandas()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].barh(type_pdf['Тип_заведения'][:10], 
                 type_pdf['Количество'][:10], 
                 color='forestgreen')
axes[0, 0].set_title('Топ-10 типов заведений по количеству')
axes[0, 0].set_xlabel('Количество заведений')
axes[0, 0].invert_yaxis()

axes[0, 1].barh(type_pdf['Тип_заведения'][:10], 
                 type_pdf['Среднее_посадочных_мест'][:10], 
                 color='coral')
axes[0, 1].set_title('Средняя вместимость по типам')
axes[0, 1].set_xlabel('Среднее посадочных мест')
axes[0, 1].invert_yaxis()

axes[1, 0].barh(type_pdf['Тип_заведения'][:10], 
                 type_pdf['Всего_посадочных_мест'][:10], 
                 color='goldenrod')
axes[1, 0].set_title('Общая вместимость по типам')
axes[1, 0].set_xlabel('Всего посадочных мест')
axes[1, 0].invert_yaxis()

for i, row in type_pdf[:10].iterrows():
    axes[1, 1].plot([row['Минимальное_мест'], row['Максимальное_мест']], 
                    [i, i], 'o-', color='purple')
axes[1, 1].set_yticks(range(10))
axes[1, 1].set_yticklabels(type_pdf['Тип_заведения'][:10])
axes[1, 1].set_title('Разброс вместимости (мин-макс)')
axes[1, 1].set_xlabel('Количество мест')
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig('анализ_по_типам.png', dpi=300, bbox_inches='tight')
plt.show()



print("\n" + "="*80)
print("УПРАЖНЕНИЕ 4: АНАЛИЗ СЕТЕВЫХ И НЕСЕТЕВЫХ ЗАВЕДЕНИЙ")
print("="*80)

network_analysis = spark.sql("""
SELECT 
    CASE 
        WHEN IsNetObject = 'да' THEN 'Сетевое'
        WHEN IsNetObject = 'нет' THEN 'Несетевое'
        ELSE 'Не указано'
    END as network_type,
    COUNT(*) as count_establishments,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM food_facilities), 2) as percentage,
    ROUND(AVG(CAST(SeatsCount AS INT)), 1) as avg_seats,
    SUM(CAST(SeatsCount AS INT)) as total_seats
FROM food_facilities
GROUP BY IsNetObject
ORDER BY count_establishments DESC
""")

print("Сравнение сетевых и несетевых заведений:")
network_analysis = network_analysis.withColumnRenamed("network_type", "Тип_сети") \
    .withColumnRenamed("count_establishments", "Количество_заведений") \
    .withColumnRenamed("percentage", "Процент") \
    .withColumnRenamed("avg_seats", "Среднее_посадочных_мест") \
    .withColumnRenamed("total_seats", "Всего_посадочных_мест")
network_analysis.show(truncate=False)


network_by_district = spark.sql("""
SELECT 
    AdmArea as adm_area,
    SUM(CASE WHEN IsNetObject = 'да' THEN 1 ELSE 0 END) as network_count,
    SUM(CASE WHEN IsNetObject = 'нет' THEN 1 ELSE 0 END) as non_network_count,
    ROUND(SUM(CASE WHEN IsNetObject = 'да' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as network_percentage
FROM food_facilities
WHERE AdmArea IS NOT NULL
GROUP BY AdmArea
HAVING COUNT(*) >= 20
ORDER BY network_percentage DESC
""")

print("\nРаспределение сетевых заведений по округам:")
network_by_district = network_by_district.withColumnRenamed("adm_area", "Административный_округ") \
    .withColumnRenamed("network_count", "Сетевые") \
    .withColumnRenamed("non_network_count", "Несетевые") \
    .withColumnRenamed("network_percentage", "Процент_сетевых")
network_by_district.show(truncate=False)

network_pdf = network_analysis.toPandas()
district_network_pdf = network_by_district.toPandas()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].pie(network_pdf['Количество_заведений'], 
            labels=network_pdf['Тип_сети'], 
            autopct='%1.1f%%',
            colors=['lightblue', 'lightcoral', 'lightgray'])
axes[0].set_title('Распределение заведений по типу сети')

bars = axes[1].barh(district_network_pdf['Административный_округ'], 
                    district_network_pdf['Процент_сетевых'],
                    color='mediumseagreen')
axes[1].set_xlabel('Процент сетевых заведений (%)')
axes[1].set_title('Доля сетевых заведений по округам')
axes[1].invert_yaxis()

for bar in bars:
    width = bar.get_width()
    axes[1].text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}%', ha='left', va='center')

plt.tight_layout()
plt.savefig('анализ_сетевых.png', dpi=300, bbox_inches='tight')
plt.show()



print("\n" + "="*80)
print("УПРАЖНЕНИЕ 5: ГЕОГРАФИЧЕСКОЕ РАСПРЕДЕЛЕНИЕ ЗАВЕДЕНИЙ")
print("="*80)

geo_analysis = spark.sql("""
SELECT 
    District as district,
    COUNT(*) as count_establishments,
    ROUND(AVG(CAST(SeatsCount AS INT)), 1) as avg_seats,
    ROUND(AVG(CAST(Latitude_WGS84 AS DOUBLE)), 6) as avg_latitude,
    ROUND(AVG(CAST(Longitude_WGS84 AS DOUBLE)), 6) as avg_longitude
FROM food_facilities
WHERE District IS NOT NULL 
    AND Latitude_WGS84 IS NOT NULL 
    AND Longitude_WGS84 IS NOT NULL
GROUP BY District
HAVING COUNT(*) >= 5
ORDER BY count_establishments DESC
LIMIT 15
""")

print("Топ-15 районов по количеству заведений:")
geo_analysis = geo_analysis.withColumnRenamed("district", "Район") \
    .withColumnRenamed("count_establishments", "Количество_заведений") \
    .withColumnRenamed("avg_seats", "Среднее_посадочных_мест") \
    .withColumnRenamed("avg_latitude", "Средняя_широта") \
    .withColumnRenamed("avg_longitude", "Средняя_долгота")
geo_analysis.show(truncate=False)


geo_data = spark.sql("""
SELECT 
    CAST(Longitude_WGS84 AS DOUBLE) as longitude,
    CAST(Latitude_WGS84 AS DOUBLE) as latitude,
    CAST(SeatsCount AS INT) as seats,
    TypeObject as type_object,
    AdmArea as area
FROM food_facilities
WHERE Longitude_WGS84 IS NOT NULL 
    AND Latitude_WGS84 IS NOT NULL 
    AND SeatsCount IS NOT NULL
    AND CAST(SeatsCount AS INT) BETWEEN 1 AND 500
""").toPandas()

print(f"\nДля визуализации отобрано {len(geo_data):,} заведений с валидными координатами")

geo_data = geo_data.rename(columns={
    'longitude': 'Долгота',
    'latitude': 'Широта',
    'seats': 'Посадочные_места',
    'type_object': 'Тип_заведения',
    'area': 'Округ'
})


plt.figure(figsize=(14, 8))

scatter = plt.scatter(geo_data['Долгота'], 
                      geo_data['Широта'], 
                      c=geo_data['Посадочные_места'],
                      cmap='viridis',
                      alpha=0.6,
                      s=30,
                      edgecolors='black',
                      linewidth=0.3)

plt.colorbar(scatter, label='Количество посадочных мест')
plt.xlabel('Долгота (WGS-84)')
plt.ylabel('Широта (WGS-84)')
plt.title('Географическое распределение заведений общественного питания в Москве')

annotations = {
    'Центральный': (37.6176, 55.7558),
    'Северо-Восточный': (37.6176, 55.8378),
    'Юго-Западный': (37.5735, 55.6704)
}

for district, (lon, lat) in annotations.items():
    plt.annotate(district, (lon, lat), 
                 fontsize=9, 
                 ha='center',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('географическое_распределение.png', dpi=300, bbox_inches='tight')
plt.show()

spark.stop()