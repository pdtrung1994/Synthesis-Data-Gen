# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from config import RESULTS_DIR

def plot_performance_comparison(results_df, dataset_name):
    """
    Draw a graph comparing the performance of different models on different datasets.
    results_df should have columns: ['Method', 'Model', 'Accuracy']
    """
    if results_df.empty:
        print(f"No results to plot for {dataset_name}.")
        return

    plt.figure(figsize=(14, 7))
    sns.barplot(x='Model', y='Accuracy', hue='Method', data=results_df)
    plt.title(f"Performance Comparison - {dataset_name}")
    plt.ylim(0, 1.0)
    
    # Place legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    
    save_path = os.path.join(RESULTS_DIR, f"{dataset_name}_performance.png")
    try:
        plt.savefig(save_path)
        print(f"Graph saved to {save_path}")
    except PermissionError:
        print(f"      [!] PermissionError: Cannot save plot {os.path.basename(save_path)}. Ensure the file is not opened in another program!")
    finally:
        plt.close()

def save_results_table(results_df, dataset_name):
    """
    Save the entire results dataframe as a CSV table.
    """
    if results_df.empty:
        return
        
    save_path = os.path.join(RESULTS_DIR, f"{dataset_name}_results.csv")
    try:
        results_df.to_csv(save_path, index=False)
        print(f"      [+] Saved full table to {os.path.basename(save_path)}")
    except PermissionError:
        print(f"      [!] PermissionError: Cannot save entire table to {os.path.basename(save_path)}. File might be open in Excel/OneDrive!")
    except Exception as e:
        print(f"      [!] Error while saving {os.path.basename(save_path)}: {e}")

def append_result_to_csv(res_dict, dataset_name):
    """
    Appends a single result row to the CSV table.
    """
    if not res_dict:
        return
        
    import time
    
    save_path = os.path.join(RESULTS_DIR, f"{dataset_name}_results.csv")
    df = pd.DataFrame([res_dict])
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            header = not os.path.exists(save_path)
            mode = 'a' if not header else 'w'
            df.to_csv(save_path, mode=mode, header=header, index=False)
            break  # Success
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait 1 second before retrying
            else:
                print(f"      [!] PermissionError: Skipping this row because {os.path.basename(save_path)} is locked by another process (OneDrive/Excel)!")
        except Exception as e:
            print(f"      [!] Error while saving row to {os.path.basename(save_path)}: {e}")
            break
