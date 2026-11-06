import os
import sys
import numpy as np
import pandas as pd

def generate_synthetic_data(filename="instacart_clean.csv", n_samples=100000, n_features=7):
    """
    如果用戶還沒下載 Kaggle 原始檔案，先自動生成符合 Instacart 格式的合成資料供立即測試編譯與執行
    特徵欄位：
    0: label (reordered: 0 or 1)
    1: order_number (用戶累計下單數)
    2: order_dow (星期幾: 0~6)
    3: order_hour_of_day (時段: 0~23)
    4: days_since_prior_order (間隔天數: 0~30)
    5: add_to_cart_order (加購物車順序: 1~30)
    6: aisle_id (貨架分類: 1~134)
    7: department_id (部門分類: 1~21)
    """
    print(f"[*] 正在生成 {n_samples} 筆符合 Instacart 格式的測試資料...")
    np.random.seed(42)
    
    order_number = np.random.randint(1, 100, size=n_samples)
    order_dow = np.random.randint(0, 7, size=n_samples)
    order_hour = np.random.randint(0, 24, size=n_samples)
    days_since = np.random.randint(0, 31, size=n_samples)
    add_to_cart = np.random.randint(1, 30, size=n_samples)
    aisle = np.random.randint(1, 135, size=n_samples)
    department = np.random.randint(1, 22, size=n_samples)
    
    # 模擬真實購物規律：加購物車順序小且間隔接近 7 天的更容易回購
    prob = 0.3 + 0.3 * (add_to_cart <= 3) + 0.2 * (days_since == 7)
    prob = np.clip(prob, 0.05, 0.95)
    reordered = (np.random.rand(n_samples) < prob).astype(int)
    
    df = pd.DataFrame({
        'reordered': reordered,
        'order_number': order_number,
        'order_dow': order_dow,
        'order_hour_of_day': order_hour,
        'days_since_prior_order': days_since,
        'add_to_cart_order': add_to_cart,
        'aisle_id': aisle,
        'department_id': department
    })
    
    df.to_csv(filename, index=False)
    print(f"[+] 合成資料生成完畢：{filename} (筆數: {n_samples}, 特徵數: {n_features})")

def process_kaggle_data(orders_path='orders.csv', 
                        order_products_path='order_products__train.csv', 
                        products_path='products.csv',
                        output_path='instacart_clean.csv'):
    """處理真實 Kaggle 下載檔案"""
    print(f"[*] 正在讀取並合併 Kaggle Instacart 資料表...")
    orders = pd.read_csv(orders_path)
    order_products = pd.read_csv(order_products_path)
    products = pd.read_csv(products_path)
    
    df = order_products.merge(orders, on='order_id', how='inner')
    df = df.merge(products[['product_id', 'aisle_id', 'department_id']], on='product_id', how='left')
    
    features = [
        'order_number', 'order_dow', 'order_hour_of_day', 
        'days_since_prior_order', 'add_to_cart_order', 
        'aisle_id', 'department_id'
    ]
    df['days_since_prior_order'] = df['days_since_prior_order'].fillna(-1)
    
    final_df = df[['reordered'] + features].astype(float)
    final_df.to_csv(output_path, index=False)
    print(f"[+] 真實資料處理完畢：{output_path} (共 {len(final_df)} 筆資料)")

if __name__ == '__main__':
    if os.path.exists('orders.csv') and os.path.exists('order_products__train.csv'):
        process_kaggle_data()
    else:
        # 如果還沒下載，預設產生 20 萬筆合成資料供開發與編譯驗證
        generate_synthetic_data('instacart_clean.csv', n_samples=200000)
