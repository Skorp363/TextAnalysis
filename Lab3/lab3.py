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

# Упражнение 1: Описание набора данных
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

# Упражнение 2: Распределение заведений по административным округам
print("\n" + "="*80)
print("УПРАЖНЕНИЕ 2: РАСПРЕДЕЛЕНИЕ ЗАВЕДЕНИЙ ПО АДМИНИСТРАТИВНЫМ ОКРУГАМ")
print("="*80)

district_distribution = spark.sql("""
SELECT 
    AdmArea as adm_area,
    COUNT(*) as count_establishments
FROM food_facilities
WHERE AdmArea IS NOT NULL
GROUP BY AdmArea
ORDER BY count_establishments DESC
""")

print("Распределение заведений по административным округам:")
district_distribution = district_distribution.withColumnRenamed("adm_area", "Административный_округ") \
    .withColumnRenamed("count_establishments", "Количество_заведений")
district_distribution.show(truncate=False)

district_pdf = district_distribution.toPandas()

plt.figure(figsize=(10, 6))
bars = plt.barh(district_pdf['Административный_округ'], 
                district_pdf['Количество_заведений'], 
                color='steelblue')
plt.xlabel('Количество заведений')
plt.title('Распределение заведений по административным округам Москвы')
plt.gca().invert_yaxis()

for bar in bars:
    width = bar.get_width()
    plt.gca().text(width + 1, bar.get_y() + bar.get_height()/2, 
                   f'{int(width)}', ha='left', va='center')

plt.tight_layout()
plt.savefig('распределение_по_округам.png', dpi=300, bbox_inches='tight')
plt.show()

# Упражнение 3: Анализ по типам заведений
print("\n" + "="*80)
print("УПРАЖНЕНИЕ 3: АНАЛИЗ ПО ТИПАМ ЗАВЕДЕНИЙ")
print("="*80)

type_analysis = spark.sql("""
SELECT 
    TypeObject as type_object,
    COUNT(*) as count
FROM food_facilities
WHERE TypeObject IS NOT NULL
GROUP BY TypeObject
HAVING COUNT(*) >= 10
ORDER BY count DESC
""")

print("Статистика по типам заведений:")
type_analysis = type_analysis.withColumnRenamed("type_object", "Тип_заведения") \
    .withColumnRenamed("count", "Количество")
type_analysis.show(truncate=False)

type_pdf = type_analysis.toPandas()

plt.figure(figsize=(10, 6))
bars = plt.barh(type_pdf['Тип_заведения'][:10], 
                type_pdf['Количество'][:10], 
                color='forestgreen')
plt.xlabel('Количество заведений')
plt.title('Топ-10 типов заведений общественного питания в Москве')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig('топ_типов_заведений.png', dpi=300, bbox_inches='tight')
plt.show()

# Упражнение 4: Анализ сетевых заведений
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
    COUNT(*) as count_establishments
FROM food_facilities
GROUP BY IsNetObject
ORDER BY count_establishments DESC
""")

print("Сравнение сетевых и несетевых заведений:")
network_analysis = network_analysis.withColumnRenamed("network_type", "Тип_сети") \
    .withColumnRenamed("count_establishments", "Количество_заведений")
network_analysis.show(truncate=False)

network_pdf = network_analysis.toPandas()

plt.figure(figsize=(8, 6))
plt.pie(network_pdf['Количество_заведений'], 
        labels=network_pdf['Тип_сети'], 
        autopct='%1.1f%%',
        colors=['lightblue', 'lightcoral', 'lightgray'])
plt.title('Распределение заведений по типу сети')

plt.tight_layout()
plt.savefig('анализ_сетевых.png', dpi=300, bbox_inches='tight')
plt.show()

# Упражнение 5: Географическое распределение заведений
print("\n" + "="*80)
print("УПРАЖНЕНИЕ 5: ГЕОГРАФИЧЕСКОЕ РАСПРЕДЕЛЕНИЕ ЗАВЕДЕНИЙ")
print("="*80)

geo_data = spark.sql("""
SELECT 
    CAST(Longitude_WGS84 AS DOUBLE) as longitude,
    CAST(Latitude_WGS84 AS DOUBLE) as latitude,
    CAST(SeatsCount AS INT) as seats
FROM food_facilities
WHERE Longitude_WGS84 IS NOT NULL 
    AND Latitude_WGS84 IS NOT NULL 
    AND SeatsCount IS NOT NULL
    AND CAST(SeatsCount AS INT) BETWEEN 1 AND 500
""").toPandas()

print(f"Для визуализации отобрано {len(geo_data):,} заведений с валидными координатами")

geo_data = geo_data.rename(columns={
    'longitude': 'Долгота',
    'latitude': 'Широта',
    'seats': 'Посадочные_места'
})

plt.figure(figsize=(10, 8))
scatter = plt.scatter(geo_data['Долгота'], 
                      geo_data['Широта'], 
                      c=geo_data['Посадочные_места'],
                      cmap='viridis',
                      alpha=0.6,
                      s=20,
                      edgecolors='black',
                      linewidth=0.3)

plt.colorbar(scatter, label='Количество посадочных мест')
plt.xlabel('Долгота (WGS-84)')
plt.ylabel('Широта (WGS-84)')
plt.title('Географическое распределение заведений общественного питания в Москве')

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('географическое_распределение.png', dpi=300, bbox_inches='tight')
plt.show()

spark.stop()

print("\n" + "="*80)
print("АНАЛИЗ ЗАВЕРШЕН. РЕЗУЛЬТАТЫ СОХРАНЕНЫ В ВИДЕ ГРАФИКОВ.")
print("="*80)