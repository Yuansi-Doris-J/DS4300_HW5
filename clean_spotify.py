import os
import pandas as pd
import re

"""
Spotify Data Preprocessing for Neo4j Import
============================================
This program performs:
1. Data reading with stratified sampling by genre (max 300 per genre, total 15000 records)
2. Data cleaning and validation
3. Handle null values in album_name and other fields
4. Format conversion for Neo4j compatibility
5. CSV format fixing to avoid Neo4j import errors
6. Genre statistics generation for reference
"""

# Configuration
CHUNK_SIZE = 50000
TARGET_PER_GENRE = 300
TOTAL_TARGET = 15000
INPUT_FILE = 'spotify.csv'
OUTPUT_FILE = 'spotify_neo4j_ready.csv'
STATS_FILE = 'genre_statistics.csv'


def clean_text_field(text):
    """
    Clean text fields to be Neo4j CSV compatible
    Remove quotes, special characters, and fix formatting issues

    Args:
        text: Input text value

    Returns:
        Cleaned text string, returns 'Unknown' for empty values
    """
    if pd.isna(text):
        return 'Unknown'

    text = str(text)

    # Handle empty or null strings
    if text == '' or text.lower() in ['nan', 'null', 'none']:
        return 'Unknown'

    # Remove double quotes
    text = text.replace('"', '')
    # Remove single quotes
    text = text.replace("'", '')
    # Replace newlines with spaces
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    # Replace commas with semicolons to avoid CSV parsing issues
    text = text.replace(',', ';')
    # Remove backslashes
    text = text.replace('\\', '/')
    # Remove any remaining special characters (keep alphanumeric, spaces, hyphens, parentheses, brackets, slashes)
    text = re.sub(r'[^\w\s\-\;\.\(\)\[\]\{\}\/]', '', text)
    # Strip leading/trailing spaces
    text = text.strip()

    # If after cleaning the text becomes empty, return 'Unknown'
    if text == '':
        return 'Unknown'

    return text


def safe_int_conversion(value, default=0):
    """
    Safely convert value to integer with default fallback

    Args:
        value: Input value to convert
        default: Default value if conversion fails

    Returns:
        Integer value or default
    """
    if pd.isna(value):
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def safe_float_conversion(value, default=0.0):
    """
    Safely convert value to float with default fallback

    Args:
        value: Input value to convert
        default: Default value if conversion fails

    Returns:
        Float value or default
    """
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_bool_conversion(value, default=False):
    """
    Safely convert value to boolean with default fallback

    Args:
        value: Input value to convert
        default: Default value if conversion fails

    Returns:
        Boolean value or default
    """
    if pd.isna(value):
        return default
    try:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ['true', '1', 'yes']
        return bool(value)
    except (ValueError, TypeError):
        return default


def read_and_sample():
    """Read CSV in chunks and perform stratified sampling"""
    print("Starting data sampling...")

    sampled_data = []
    genres_count = {}

    print(f"Collecting up to {TARGET_PER_GENRE} records per genre...")

    for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE):
        # Filter out low popularity records
        chunk = chunk[chunk['popularity'] > 0].copy()

        if len(chunk) == 0:
            continue

        for genre, group in chunk.groupby('track_genre'):
            if genre not in genres_count:
                genres_count[genre] = 0

            needed = TARGET_PER_GENRE - genres_count[genre]
            if needed > 0:
                take = min(needed, len(group))
                sampled_data.append(group.head(take))
                genres_count[genre] += take

        current_total = sum(genres_count.values())
        if current_total >= TOTAL_TARGET:
            print(f"  Reached target, stopping...")
            break

    # Combine all sampled data
    df = pd.concat(sampled_data, ignore_index=True)
    print(f"Sampling completed: {len(df)} records, {df['track_genre'].nunique()} genres")

    return df


def clean_data(df):
    """Clean and validate data for Neo4j import"""
    print("\nStarting data cleaning...")

    # Create a fresh copy to avoid warnings
    df = df.copy()

    # Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates(subset=['track_id'])
    print(f"Removed {initial_count - len(df)} duplicate track_id(s)")

    # Handle duration_ms with safe conversion
    df['duration_ms'] = df['duration_ms'].apply(lambda x: safe_int_conversion(x, 0))
    df = df[df['duration_ms'] > 0].copy()
    print(f"Records after duration filter: {len(df)}")

    # Handle missing values for numeric columns with safe conversion
    numeric_cols = ['danceability', 'energy', 'loudness', 'speechiness',
                    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: safe_float_conversion(x, 0.0))

    # Handle key and mode
    df['key'] = df['key'].apply(lambda x: safe_int_conversion(x, -1))
    df['mode'] = df['mode'].apply(lambda x: safe_int_conversion(x, 0))

    # Handle explicit flag
    df['explicit'] = df['explicit'].apply(lambda x: safe_bool_conversion(x, False))

    print(f"Cleaning completed: {len(df)} records")
    return df


def handle_null_values(df):
    """Handle null values in album_name and other fields"""
    print("\nHandling null values...")

    # Create a fresh copy to avoid warnings
    df = df.copy()

    # Clean text fields with the enhanced cleaner (will return 'Unknown' for empty values)
    text_cols = ['artists', 'album_name', 'track_name', 'track_genre']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_text_field)

    # Clip popularity to valid range
    df['popularity'] = df['popularity'].clip(0, 100)

    # Final check: ensure no null values in critical fields
    print("\nChecking for null values after cleaning:")
    for col in text_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            print(f"  Warning: {col} has {null_count} null values - filling with 'Unknown'")
            df[col] = df[col].fillna('Unknown')

    print(f"Null value handling completed")
    return df


def fix_csv_format(df, output_file):
    """
    Save DataFrame to CSV with Neo4j-compatible formatting
    Use minimal quoting to avoid parsing issues

    Args:
        df: DataFrame to save
        output_file: Output file path
    """
    print("\nSaving CSV with Neo4j-compatible format...")

    # Save with minimal quoting
    try:
        # For newer pandas versions (>= 1.5.0)
        df.to_csv(output_file,
                  index=False,
                  encoding='utf-8',
                  quoting=0,  # QUOTE_MINIMAL - only quote when needed
                  escapechar='\\',
                  doublequote=False,
                  lineterminator='\n')
    except TypeError:
        # Fallback for older pandas versions
        df.to_csv(output_file,
                  index=False,
                  encoding='utf-8',
                  quoting=0,
                  escapechar='\\',
                  doublequote=False,
                  line_terminator='\n')

    # Post-process to fix any remaining issues
    with open(output_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Fix any remaining quote issues
    content = content.replace('""', '"')
    content = content.replace('""SS', 'SS')
    content = re.sub(r'"([^"]*)"([^"]*)"', r'"\1\2"', content)

    # Ensure no empty fields are left (replace empty fields with 'Unknown')
    lines = content.split('\n')
    header = lines[0].split(',')
    cleaned_lines = [lines[0]]  # Keep header

    for line in lines[1:]:
        if line.strip():
            fields = line.split(',')
            # Ensure all fields have values, replace empty with 'Unknown'
            for i, field in enumerate(fields):
                if field == '' or field == '""':
                    fields[i] = 'Unknown'
            cleaned_lines.append(','.join(fields))

    content = '\n'.join(cleaned_lines)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"CSV saved with Neo4j-compatible formatting")


def format_for_neo4j(df):
    """Format data for Neo4j CSV import"""
    print("\nFormatting for Neo4j import...")

    # Create a copy to avoid warnings
    df = df.copy()

    # Reorder columns for better readability in Neo4j
    column_order = [
        'track_id', 'track_name', 'artists', 'album_name', 'track_genre',
        'popularity', 'duration_ms', 'explicit',
        'danceability', 'energy', 'acousticness', 'valence', 'tempo',
        'loudness', 'speechiness', 'instrumentalness', 'liveness',
        'key', 'mode'
    ]

    # Ensure all columns exist
    existing_cols = [col for col in column_order if col in df.columns]
    df = df[existing_cols]

    print(f"Formatted {len(df)} records with {len(df.columns)} columns")
    return df


def generate_statistics(df):
    """Generate genre statistics for reference"""
    print("\nGenerating genre statistics...")

    # Create a copy to avoid warnings
    df = df.copy()

    stats = df.groupby('track_genre').agg({
        'track_id': 'count',
        'popularity': 'mean',
        'duration_ms': 'mean'
    }).rename(columns={
        'track_id': 'record_count',
        'popularity': 'avg_popularity',
        'duration_ms': 'avg_duration_ms'
    })
    stats = stats.sort_values('record_count', ascending=False)

    print(f"\n=== Genre Statistics ===")
    print(f"Total genres: {len(stats)}")
    print(f"Total records: {stats['record_count'].sum()}")
    print(f"\nTop 10 genres by count:")
    print(stats.head(10))
    print(f"\nBottom 10 genres by count:")
    print(stats.tail(10))

    return stats


def verify_csv_for_neo4j(file_path):
    """
    Verify CSV file format for Neo4j import
    """
    print(f"\nVerifying CSV file: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            print(f"Header: {first_line.strip()[:100]}...")

            # Check first 50 rows for issues
            issue_count = 0
            empty_album_count = 0

            for i, line in enumerate(f):
                if i >= 50:
                    break

                fields = line.strip().split(',')

                # Check for album_name field (index 3)
                if len(fields) > 3:
                    album_name = fields[3].strip('"')
                    if album_name == '' or album_name.lower() in ['', 'unknown', 'null', 'nan']:
                        empty_album_count += 1
                        print(f"  Info: Row {i + 2} has empty album_name")

                # Check for unbalanced quotes
                quote_count = line.count('"')
                if quote_count % 2 != 0:
                    print(f"  Warning: Row {i + 2} has unbalanced quotes")
                    issue_count += 1

                # Check for problematic patterns
                if '""SS' in line:
                    print(f"  Warning: Row {i + 2} contains '""SS' pattern")
                    issue_count += 1

            if empty_album_count > 0:
                print(f"\nFound {empty_album_count} rows with empty album_name in first 50 rows")
                print("These will be handled by the Neo4j import using COALESCE")

            if issue_count == 0:
                print("\nCSV format verification passed!")
            else:
                print(f"\nFound {issue_count} potential issues")

    except Exception as e:
        print(f"Error verifying CSV: {e}")

    return issue_count == 0


def main():
    """Main execution function"""
    print("=" * 50)
    print("Spotify Data Preprocessing for Neo4j")
    print("=" * 50)

    # Step 1: Data reading with stratified sampling by genre
    df = read_and_sample()

    # Step 2: Data cleaning and validation
    df = clean_data(df)

    # Step 3: Handle null values in album_name and other fields
    df = handle_null_values(df)

    # Step 4: Format conversion for Neo4j compatibility
    df = format_for_neo4j(df)

    # Step 5: CSV format fixing to avoid Neo4j import errors
    print(f"\nSaving files...")
    fix_csv_format(df, OUTPUT_FILE)

    # Step 6: Genre statistics generation for reference
    stats = generate_statistics(df)
    stats.to_csv(STATS_FILE, index=False, encoding='utf-8')

    # Verify the output file
    verify_csv_for_neo4j(OUTPUT_FILE)

    print(f"\nOutput files created:")
    print(f"  - {OUTPUT_FILE}: {len(df)} records ready for Neo4j import")
    print(f"  - {STATS_FILE}: genre statistics for reference")

    # Print file size
    file_size = os.path.getsize(OUTPUT_FILE) / 1024
    print(f"  - File size: {file_size:.1f} KB")

    # Print sample for verification
    print(f"\nSample record (first row):")
    if len(df) > 0:
        sample_row = df.iloc[0].to_dict()
        for i, (key, value) in enumerate(list(sample_row.items())[:10]):
            print(f"  {key}: {value}")

    print("\n" + "=" * 50)
    print("Preprocessing complete!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Copy the CSV file to Neo4j import directory:")
    print(f"   copy {OUTPUT_FILE} C:\\neo4j\\import\\")
    print("2. Run Neo4j import commands (use the Neo4j Browser)")
    print("3. For Album import, use COALESCE to handle null values:")
    print("   MERGE (al:Album {name: COALESCE(row.album_name, 'Unknown Album')})")
    print("4. Verify the import with: MATCH (n) RETURN labels(n), COUNT(n)")


if __name__ == "__main__":
    main()
