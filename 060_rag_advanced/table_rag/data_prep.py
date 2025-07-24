#%% create sqlite database
import sqlite3

#%%
sql_create_table = """
-- Table 1: Global Coffee Sales
CREATE TABLE coffee_sales (
    sale_id INT PRIMARY KEY,
    product_name VARCHAR(100),
    origin_country VARCHAR(50),
    bean_type VARCHAR(50),
    roast_level VARCHAR(20),
    price_per_kg DECIMAL(6,2),
    quantity_kg DECIMAL(6,2),
    sale_date DATE,
    customer_id INT,
    region VARCHAR(50),
    organic BOOLEAN,
    certification VARCHAR(50)
);
"""
sql_insert_data = """

-- Insert sample data
INSERT INTO coffee_sales VALUES
(1, 'Ethiopian Yirgacheffe', 'Ethiopia', 'Arabica', 'Light', 28.50, 500.00, '2024-08-15', 103, 'North America', true, 'Fair Trade'),
(2, 'Colombian Supremo', 'Colombia', 'Arabica', 'Medium', 24.75, 750.00, '2024-07-22', 205, 'Europe', true, 'Rainforest Alliance'),
(3, 'Sumatra Mandheling', 'Indonesia', 'Arabica', 'Dark', 22.00, 300.00, '2024-09-01', 187, 'Asia', false, NULL),
(4, 'Jamaican Blue Mountain', 'Jamaica', 'Arabica', 'Medium', 85.00, 100.00, '2024-08-05', 321, 'North America', true, 'Organic'),
(5, 'Brazilian Santos', 'Brazil', 'Arabica', 'Medium-Dark', 18.50, 1200.00, '2024-09-12', 142, 'South America', false, NULL),
(6, 'Vietnamese Robusta', 'Vietnam', 'Robusta', 'Dark', 15.75, 2000.00, '2024-07-30', 256, 'Asia', false, NULL),
(7, 'Kenyan AA', 'Kenya', 'Arabica', 'Medium', 32.00, 400.00, '2024-08-22', 119, 'Europe', true, 'Fair Trade'),
(8, 'Costa Rican Tarrazu', 'Costa Rica', 'Arabica', 'Light-Medium', 27.50, 350.00, '2024-09-05', 244, 'North America', true, 'Bird Friendly'),
(9, 'Guatemalan Antigua', 'Guatemala', 'Arabica', 'Medium', 26.25, 600.00, '2024-08-18', 173, 'Europe', false, 'Rainforest Alliance'),
(10, 'Indian Monsooned Malabar', 'India', 'Arabica', 'Medium-Dark', 24.00, 300.00, '2024-09-10', 298, 'Asia', false, NULL);
"""



# %% create database
conn = sqlite3.connect('coffee_sales.db')
cursor = conn.cursor()
cursor.execute(sql_create_table)
cursor.execute(sql_insert_data)
conn.commit()
conn.close()


# %%
