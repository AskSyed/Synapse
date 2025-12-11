from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Optional

app = FastAPI()

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    in_stock: bool = True

items_db: Dict[int, Item] = {
    1: Item(name="Laptop", description="A high-end gaming laptop", price=1500.00),
    2: Item(name="Smartphone", description="Latest model smartphone", price=800.00, in_stock=False),
}
next_item = 3

@app.get("/items")
def get_all_items():
    return {
        "total": len(items_db),
        "items": items_db
    }

@app.post("/items")
def create_item(item: Item):
    global next_item
    items_db[next_item] = item
    next_item += 1
    return {"message": "Item created successfully"}

@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    if item_id in items_db:
        items_db[item_id] = item
        return {"message": "Item updated successfully"}
    return {"error": "Item not found"}

@app.delete("/items/{item_id}")
def delete_item(item_id: int):
    if item_id in items_db:
        del items_db[item_id]
        return {"message": "Item deleted successfully"}
    return {"error": "Item not found"}
