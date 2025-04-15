import pandas as pd
import hashlib
from io import StringIO
    
def hash_dataframe(
    df: pd.DataFrame,
    sort_by = None,
    debug: bool = False
    ) -> str:
    """Converts dataframe to a normalized hash string for comparison."""
    
    df = df.copy()

    # Optional stable sort
    if sort_by:
        df = df.sort_values(by=sort_by).reset_index(drop=True)
    else:
        df = df.sort_index()
    
    # Sort the index and columns to ensure consistent order
    df = df[sorted(df.columns)]
    
    # Convert all columns to string type for consistent hashing
    df.index = pd.Index(df.index, dtype="int64")
    
    # Write to CSV with fixed settings
    buffer = StringIO()
    df.to_csv(
        buffer,
        index=True,
        line_terminator="\n",
        float_format="%.10f",
        na_rep="NaN"
    )
    
    if debug:
        # Print the CSV string for debugging purposes
        print("CSV String:")
        print(buffer.getvalue())
    
    return hashlib.md5(buffer.getvalue().encode('utf-8')).hexdigest()