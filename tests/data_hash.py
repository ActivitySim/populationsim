import pandas as pd
import hashlib
from io import StringIO
    
def hash_dataframe(df: pd.DataFrame) -> str:
    """Converts dataframe to a normalized hash string for comparison."""
    
    # Sort to ensure consistent row and column order and normalize dtypes
    df = df.sort_index(axis=0).sort_index(axis=1).convert_dtypes()

    # Write to string with fixed settings
    buffer = StringIO()
    
    # Use line_terminator="\n" to ensure consistent line endings across platforms
    df.to_csv(buffer, index=True, line_terminator="\n", float_format="%.10f")
    
    return hashlib.md5(buffer.getvalue().encode('utf-8')).hexdigest()