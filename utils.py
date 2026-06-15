from typing import List, Dict

def displayTabSummary(farmsData: List[Dict[str, str]]):
    header = f"{'Farm ID':<12} | {'Crop Type':<12} | {'Geographical Boundary':<30}"

    div = "-" * len(header)

    print(div + "\n" + header + "\n" + div)

    for row in farmsData:
        print(f"{row['id']:<12} | {row['crop']:<12} | {row['bounds']:<30}")

    print(div + "\n")

# displayTabSummary() 