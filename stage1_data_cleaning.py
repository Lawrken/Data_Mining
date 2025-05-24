# Netflix Dataset Analysis - Stage 1: Data Cleaning
# Data Mining Final Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set display options for better data viewing
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

print("=== NETFLIX DATASET DATA CLEANING ===")
print("Stage 1 of Data Mining Project\n")

# 1. LOAD THE DATASET
print("1. LOADING DATASET...")
# Replace 'netflix_titles.csv' with your actual file path
try:
    df = pd.read_csv('netflix_titles.csv')
    print(f"✓ Dataset loaded successfully!")
    print(f"  - Shape: {df.shape}")
    print(f"  - Columns: {len(df.columns)}")
except FileNotFoundError:
    print("❌ Error: Please ensure 'netflix_titles.csv' is in the same directory")
    print("   or update the file path in the code above")

# 2. INITIAL DATA EXPLORATION
print("\n2. INITIAL DATA EXPLORATION...")
print("\nDataset Info:")
print(df.info())

print("\nFirst few rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset shape:", df.shape)

# 3. MISSING VALUES ANALYSIS
print("\n3. MISSING VALUES ANALYSIS...")
missing_data = df.isnull().sum()
missing_percentage = (missing_data / len(df)) * 100

missing_summary = pd.DataFrame({
    'Missing_Count': missing_data,
    'Missing_Percentage': missing_percentage
}).sort_values('Missing_Percentage', ascending=False)

print("Missing values summary:")
print(missing_summary[missing_summary['Missing_Count'] > 0])

# Visualize missing data
if len(missing_summary[missing_summary['Missing_Count'] > 0]) > 0:
    plt.figure(figsize=(12, 6))
    missing_summary[missing_summary['Missing_Count'] > 0].plot(kind='bar', y='Missing_Percentage')
    plt.title('Missing Data Percentage by Column')
    plt.ylabel('Percentage Missing')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Try to display the plot
    try:
        plt.show(block=False)  # Non-blocking show
        print("📊 Missing data visualization displayed as Figure 1")
    except Exception as e:
        print(f"⚠️ Could not display plot: {e}")
        print("📊 Missing data chart data:")
        print(missing_summary[missing_summary['Missing_Count'] > 0][['Missing_Count', 'Missing_Percentage']])
    
    # Also save the plot as an image file
    try:
        plt.savefig('missing_data_chart.png', dpi=300, bbox_inches='tight')
        print("💾 Missing data chart saved as 'missing_data_chart.png'")
    except Exception as e:
        print(f"⚠️ Could not save plot: {e}")
else:
    print("✅ No missing data to visualize!")

# 4. DATA CLEANING STEPS
print("\n4. DATA CLEANING PROCESS...")

# Create a copy for cleaning
df_clean = df.copy()
print(f"Created working copy with shape: {df_clean.shape}")

# 4.1 Handle Missing Values
print("\n4.1 Handling Missing Values...")

# For categorical columns, fill with 'Unknown' or most appropriate value
categorical_columns = ['director', 'cast', 'country', 'rating']
for col in categorical_columns:
    if col in df_clean.columns:
        before_count = df_clean[col].isnull().sum()
        if col == 'director':
            df_clean[col] = df_clean[col].fillna('Unknown Director')
        elif col == 'cast':
            df_clean[col] = df_clean[col].fillna('Unknown Cast')
        elif col == 'country':
            df_clean[col] = df_clean[col].fillna('Unknown Country')
        elif col == 'rating':
            # Fill rating with the most common rating for that type
            mode_rating = df_clean['rating'].mode()[0] if not df_clean['rating'].mode().empty else 'Not Rated'
            df_clean[col] = df_clean[col].fillna(mode_rating)
        
        after_count = df_clean[col].isnull().sum()
        print(f"  - {col}: {before_count} → {after_count} missing values")

# For date columns, handle missing dates
if 'date_added' in df_clean.columns:
    before_count = df_clean['date_added'].isnull().sum()
    # Keep missing dates as NaN for now, we'll handle them in data type conversion
    after_count = df_clean['date_added'].isnull().sum()
    print(f"  - date_added: {before_count} missing values (keeping as NaN for now)")

# 4.2 Remove Duplicates
print("\n4.2 Removing Duplicates...")
before_shape = df_clean.shape
df_clean = df_clean.drop_duplicates()
after_shape = df_clean.shape
duplicates_removed = before_shape[0] - after_shape[0]
print(f"  - Removed {duplicates_removed} duplicate rows")
print(f"  - Dataset shape: {before_shape} → {after_shape}")

# 4.3 Data Type Conversions
print("\n4.3 Data Type Conversions...")

# Convert date_added to datetime
if 'date_added' in df_clean.columns:
    try:
        # Convert all date_added entries to datetime, replacing 'Unknown Date' with NaT
        df_clean['date_added'] = df_clean['date_added'].replace('Unknown Date', pd.NaT)
        df_clean['date_added'] = pd.to_datetime(df_clean['date_added'], errors='coerce')
        print("  ✓ Converted date_added to datetime")
    except Exception as e:
        print(f"  ❌ Error converting date_added: {e}")

# Ensure release_year is numeric
if 'release_year' in df_clean.columns:
    df_clean['release_year'] = pd.to_numeric(df_clean['release_year'], errors='coerce')
    print("  ✓ Ensured release_year is numeric")

# 4.4 Text Data Cleaning
print("\n4.4 Text Data Cleaning...")

# Clean and standardize text columns
text_columns = ['title', 'description', 'listed_in', 'director', 'cast', 'country']
for col in text_columns:
    if col in df_clean.columns:
        # Remove extra whitespace
        df_clean[col] = df_clean[col].astype(str).str.strip()
        # Remove multiple spaces
        df_clean[col] = df_clean[col].str.replace(r'\s+', ' ', regex=True)
        print(f"  ✓ Cleaned text formatting in {col}")

# 4.5 Create Derived Columns
print("\n4.5 Creating Derived Columns...")

# Extract year from date_added
if 'date_added' in df_clean.columns:
    # Only extract year where date_added is not null
    df_clean['year_added'] = df_clean['date_added'].dt.year.astype('Int64')  # Use nullable integer
    print("  ✓ Created year_added column")

# Create decade column from release_year
if 'release_year' in df_clean.columns:
    df_clean['release_decade'] = (df_clean['release_year'] // 10) * 10
    print("  ✓ Created release_decade column")

# Count number of countries (for content with multiple countries)
if 'country' in df_clean.columns:
    # Handle NaN values properly
    df_clean['country_count'] = df_clean['country'].str.count(',').fillna(-1) + 1
    df_clean.loc[df_clean['country'] == 'Unknown Country', 'country_count'] = 0
    df_clean.loc[df_clean['country'].str.contains('nan', case=False, na=False), 'country_count'] = 0
    print("  ✓ Created country_count column")

# Count number of genres
if 'listed_in' in df_clean.columns:
    df_clean['genre_count'] = df_clean['listed_in'].str.count(',') + 1
    print("  ✓ Created genre_count column")

# 4.6 Handle Outliers
print("\n4.6 Handling Outliers...")

# Check for unrealistic release years
if 'release_year' in df_clean.columns:
    current_year = datetime.now().year
    unrealistic_years = df_clean[(df_clean['release_year'] < 1900) | (df_clean['release_year'] > current_year)]
    print(f"  - Found {len(unrealistic_years)} entries with unrealistic release years")
    
    # You might want to investigate these further or remove them
    # For now, we'll keep them but flag them
    df_clean['year_flag'] = 'Normal'
    df_clean.loc[(df_clean['release_year'] < 1900) | (df_clean['release_year'] > current_year), 'year_flag'] = 'Questionable'

# 5. FINAL DATA VALIDATION
print("\n5. FINAL DATA VALIDATION...")

print("Final dataset info:")
print(f"  - Shape: {df_clean.shape}")
print(f"  - Columns: {len(df_clean.columns)}")

print("\nFinal missing values:")
final_missing = df_clean.isnull().sum()
print(final_missing[final_missing > 0])

print("\nData types:")
print(df_clean.dtypes)

print("\nSample of cleaned data:")
print(df_clean.head())

# 6. SAVE CLEANED DATASET
print("\n6. SAVING CLEANED DATASET...")
try:
    df_clean.to_csv('netflix_titles_cleaned.csv', index=False)
    print("✓ Cleaned dataset saved as 'netflix_titles_cleaned.csv'")
except Exception as e:
    print(f"❌ Error saving file: {e}")

# 7. CLEANING SUMMARY REPORT
print("\n" + "="*50)
print("DATA CLEANING SUMMARY REPORT")
print("="*50)
print(f"Original dataset shape: {df.shape}")
print(f"Cleaned dataset shape: {df_clean.shape}")
print(f"Rows removed: {df.shape[0] - df_clean.shape[0]}")
print(f"Duplicates removed: {duplicates_removed}")

print("\nColumns added:")
new_columns = set(df_clean.columns) - set(df.columns)
for col in new_columns:
    print(f"  - {col}")

print("\nData quality improvements:")
print("  ✓ Missing values handled")
print("  ✓ Duplicates removed")
print("  ✓ Data types optimized")
print("  ✓ Text data standardized")
print("  ✓ Derived columns created")
print("  ✓ Outliers flagged")

print("\n✅ DATA CLEANING COMPLETE!")
print("Ready for Stage 2: Data Analysis and Visualization")

# Optional: Quick preview of what's coming in Stage 2
print("\n" + "="*50)
print("PREVIEW: Stage 2 Analysis Ready")
print("="*50)
print("Your cleaned dataset is now ready for:")
print("1. Content Type Distribution Analysis")
print("2. Temporal Trends Analysis") 
print("3. Top Genres Analysis")
print("4. Geographic Distribution Analysis")
print("5. Rating Breakdown Analysis")