import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import networkx as nx
from geopy.distance import great_circle

class SupplyChainOptimizationAgent:
    def __init__(self, inventory_data, sales_data, suppliers, locations):
        self.inventory_data = inventory_data
        self.sales_data = sales_data
        self.suppliers = suppliers
        self.locations = locations
        self.model = RandomForestRegressor()
    
    def forecast_demand(self):
        self.sales_data['Month'] = pd.to_datetime(self.sales_data['Date']).dt.month
        X = self.sales_data[['Product_ID', 'Month']]
        y = self.sales_data['Sales']
        self.model.fit(X, y)
        self.sales_data['Forecast'] = self.model.predict(X)
    
    def reorder_stock(self):
        low_stock = self.inventory_data[self.inventory_data['Quantity'] < self.inventory_data['Reorder_Level']]
        orders = []
        for product in low_stock['Product_ID']:
            supplier = self.suppliers.loc[self.suppliers['Product_ID'] == product]
            orders.append({'Product_ID': product, 'Supplier_ID': supplier['Supplier_ID'].values[0], 'Order_Quantity': self.inventory_data.loc[self.inventory_data['Product_ID'] == product, 'Reorder_Quantity'].values[0]})
        return orders
    
    def optimize_routes(self, delivery_points):
        graph = nx.Graph()
        for i, loc1 in enumerate(self.locations):
            for j, loc2 in enumerate(self.locations):
                if i != j:
                    distance = great_circle(loc1, loc2).km
                    graph.add_edge(i, j, weight=distance)
        optimized_routes = nx.shortest_path(graph, source=0, target=len(delivery_points)-1, weight='weight')
        return optimized_routes
    
    def monitor_supply_chain(self):
        self.forecast_demand()
        orders = self.reorder_stock()
        routes = self.optimize_routes(self.locations)
        return {'Orders': orders, 'Routes': routes}
