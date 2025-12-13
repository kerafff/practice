
# FastAPI backend


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
from sqlalchemy import create_engine, MetaData, Table, select, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from fastapi.middleware.cors import CORSMiddleware
import math
import os


DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "123")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


engine = create_engine(DATABASE_URL)
metadata = MetaData()

try:
    
    producttype = Table("producttype", metadata, autoload_with=engine)
    materialtype = Table("materialtype", metadata, autoload_with=engine)
    workshopcategory = Table("workshopcategory", metadata, autoload_with=engine)
    workshop = Table("workshop", metadata, autoload_with=engine)
    product = Table("product", metadata, autoload_with=engine)
    productionprocess = Table("productionprocess", metadata, autoload_with=engine)
except Exception as e:
    raise RuntimeError("Ошибка отражения таблиц в БД. Убедитесь, что таблицы существуют и соединение настроено. " + str(e))

app = FastAPI(title="Furniture Production API (ТЗ)")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProductCreate(BaseModel):
    product_type_id: int = Field(..., gt=0)
    product_name: str = Field(..., min_length=2)
    article_number: str = Field(..., min_length=1)
    min_partner_price: float = Field(..., gt=0)
    material_type_id: int = Field(..., gt=0)

    @validator("product_name", "article_number", pre=True)
    def strip_strings(cls, v):
        return v.strip() if isinstance(v, str) else v

class ProductTypeModel(BaseModel):
    type_name: str = Field(..., min_length=2)
    coefficient: float = Field(..., gt=0)

    @validator("type_name", pre=True)
    def strip_name(cls, v):
        return v.strip()

class ProcessCreate(BaseModel):
    workshop_id: int = Field(..., gt=0)
    manufacturing_hours: int = Field(..., gt=0)  


def exists_in_table(tbl: Table, id_value: int) -> bool:
    pk = list(tbl.primary_key)[0].name
    stmt = select(tbl).where(tbl.c[pk] == id_value).limit(1)
    with engine.connect() as conn:
        r = conn.execute(stmt).first()
        return r is not None

def row_to_dicts(result_proxy) -> List[Dict]:
    """Convert SQLAlchemy Result to list of dicts (mappings)."""
    return [dict(r) for r in result_proxy.mappings().all()]



@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/products")
def get_products():
    q = text("""
        SELECT p.product_id, p.article_number, p.product_name,
               pt.type_name AS product_type, mt.material_name,
               p.min_partner_price
        FROM product p
        LEFT JOIN producttype pt ON p.product_type_id = pt.product_type_id
        LEFT JOIN materialtype mt ON p.material_type_id = mt.material_type_id
        ORDER BY p.product_id
    """)
    try:
        with engine.connect() as conn:
            rows = conn.execute(q)
            return row_to_dicts(rows)
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/products")
def add_product(item: ProductCreate):
    if not exists_in_table(producttype, item.product_type_id):
        raise HTTPException(status_code=400, detail="product_type_id не найден")
    if not exists_in_table(materialtype, item.material_type_id):
        raise HTTPException(status_code=400, detail="material_type_id не найден")
    insert_q = text("""
        INSERT INTO product (product_type_id, product_name, article_number, min_partner_price, material_type_id)
        VALUES (:ptype, :pname, :article, :price, :mtype)
        RETURNING product_id
    """)
    params = {"ptype": item.product_type_id, "pname": item.product_name, "article": item.article_number,
              "price": item.min_partner_price, "mtype": item.material_type_id}
    try:
        with engine.begin() as conn:
            res = conn.execute(insert_q, params)
            new_id = res.scalar_one()
        return {"status": "ok", "product_id": int(new_id)}
    except IntegrityError as ie:
        raise HTTPException(status_code=400, detail=str(ie.orig))
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/products/{product_id}")
def update_product(product_id: int, item: ProductCreate):
    if not exists_in_table(product, product_id):
        raise HTTPException(status_code=404, detail="Product not found")
    if not exists_in_table(producttype, item.product_type_id):
        raise HTTPException(status_code=400, detail="product_type_id не найден")
    if not exists_in_table(materialtype, item.material_type_id):
        raise HTTPException(status_code=400, detail="material_type_id не найден")
    update_q = text("""
        UPDATE product
        SET product_type_id = :ptype,
            product_name = :pname,
            article_number = :article,
            min_partner_price = :price,
            material_type_id = :mtype
        WHERE product_id = :pid
    """)
    params = {"ptype": item.product_type_id, "pname": item.product_name, "article": item.article_number,
              "price": item.min_partner_price, "mtype": item.material_type_id, "pid": product_id}
    try:
        with engine.begin() as conn:
            conn.execute(update_q, params)
        return {"status": "updated"}
    except IntegrityError as ie:
        raise HTTPException(status_code=400, detail=str(ie.orig))
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/products/{product_id}")
def delete_product(product_id: int):
    if not exists_in_table(product, product_id):
        raise HTTPException(status_code=404, detail="Product not found")
    try:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM productionprocess WHERE product_id = :pid"), {"pid": product_id})
            conn.execute(text("DELETE FROM product WHERE product_id = :pid"), {"pid": product_id})
        return {"status": "deleted"}
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/product-types")
def get_product_types():
    try:
        stmt = select(producttype).order_by(producttype.c.product_type_id)
        with engine.connect() as conn:
            rows = conn.execute(stmt)
            return row_to_dicts(rows)
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/product-types")
def add_product_type(item: ProductTypeModel):
    insert = text("INSERT INTO producttype (type_name, coefficient) VALUES (:name, :coef) RETURNING product_type_id")
    try:
        with engine.begin() as conn:
            res = conn.execute(insert, {"name": item.type_name, "coef": item.coefficient})
            new_id = res.scalar_one()
        return {"status": "ok", "product_type_id": int(new_id)}
    except IntegrityError:
        raise HTTPException(status_code=400, detail="Тип с таким названием уже существует")
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/product-types/{type_id}")
def update_product_type(type_id: int, item: ProductTypeModel):
    if not exists_in_table(producttype, type_id):
        raise HTTPException(status_code=404, detail="Type not found")
    try:
        with engine.begin() as conn:
            conn.execute(text("UPDATE producttype SET type_name = :name, coefficient = :coef WHERE product_type_id = :id"),
                         {"name": item.type_name, "coef": item.coefficient, "id": type_id})
        return {"status": "updated"}
    except IntegrityError:
        raise HTTPException(status_code=400, detail="Название уже используется")
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/product-types/{type_id}")
def delete_product_type(type_id: int):
    if not exists_in_table(producttype, type_id):
        raise HTTPException(status_code=404, detail="Type not found")
    with engine.connect() as conn:
        cnt = conn.execute(text("SELECT COUNT(*) FROM product WHERE product_type_id = :id"), {"id": type_id}).scalar_one()
        if cnt > 0:
            raise HTTPException(status_code=400, detail="Нельзя удалить: тип используется в продуктах")
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM producttype WHERE product_type_id = :id"), {"id": type_id})
    return {"status": "deleted"}


@app.get("/material-types")
def get_material_types():
    try:
        stmt = select(materialtype).order_by(materialtype.c.material_type_id)
        with engine.connect() as conn:
            rows = conn.execute(stmt)
            return row_to_dicts(rows)
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/workshops")
def get_workshops():
    q = text("""
        SELECT w.workshop_id, w.workshop_name, wc.category_name, w.staff_count
        FROM workshop w
        LEFT JOIN workshopcategory wc ON w.category_id = wc.category_id
        ORDER BY w.workshop_id
    """)
    try:
        with engine.connect() as conn:
            rows = conn.execute(q)
            return row_to_dicts(rows)
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/workshops_for_product/{product_id}")
def workshops_for_product(product_id: int):
    if not exists_in_table(product, product_id):
        raise HTTPException(status_code=404, detail="Product not found")
    q = text("""
        SELECT pp.process_id, pp.product_id, pp.workshop_id, pp.manufacturing_hours,
               w.workshop_name, wc.category_name, w.staff_count
        FROM productionprocess pp
        JOIN workshop w ON pp.workshop_id = w.workshop_id
        LEFT JOIN workshopcategory wc ON w.category_id = wc.category_id
        WHERE pp.product_id = :pid
        ORDER BY pp.process_id
    """)
    try:
        with engine.connect() as conn:
            rows = conn.execute(q, {"pid": product_id})
            return row_to_dicts(rows)
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/production-process/{product_id}")
def get_production_process(product_id: int):
    if not exists_in_table(product, product_id):
        raise HTTPException(status_code=404, detail="Product not found")
    q = text("""
        SELECT pp.process_id, pp.product_id, pp.workshop_id, w.workshop_name, wc.category_name, w.staff_count, pp.manufacturing_hours
        FROM productionprocess pp
        JOIN workshop w ON w.workshop_id = pp.workshop_id
        LEFT JOIN workshopcategory wc ON w.category_id = wc.category_id
        WHERE pp.product_id = :pid
        ORDER BY pp.process_id
    """)
    try:
        with engine.connect() as conn:
            rows = conn.execute(q, {"pid": product_id})
            return row_to_dicts(rows)
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/production-process/{product_id}")
def add_production_process(product_id: int, item: ProcessCreate):
    if not exists_in_table(product, product_id):
        raise HTTPException(status_code=404, detail="Product not found")
    if not exists_in_table(workshop, item.workshop_id):
        raise HTTPException(status_code=400, detail="Workshop not found")
    insert = text("""
        INSERT INTO productionprocess (product_id, workshop_id, manufacturing_hours)
        VALUES (:pid, :wid, :hours)
        RETURNING process_id
    """)
    try:
        with engine.begin() as conn:
            res = conn.execute(insert, {"pid": product_id, "wid": item.workshop_id, "hours": int(item.manufacturing_hours)})
            new_id = res.scalar_one()
        return {"process_id": int(new_id)}
    except IntegrityError:
        raise HTTPException(status_code=400, detail="Duplicate or integrity error (product_id, workshop_id)")
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/production-process/{process_id}")
def delete_production_process(process_id: int):
    stmt = select(productionprocess).where(productionprocess.c.process_id == process_id).limit(1)
    with engine.connect() as conn:
        row = conn.execute(stmt).first()
        if not row:
            raise HTTPException(status_code=404, detail="Process not found")
    try:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM productionprocess WHERE process_id = :pid"), {"pid": process_id})
        return {"status": "deleted"}
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/manufacture-time/{product_id}")
def manufacture_time(product_id: int):
    if not exists_in_table(product, product_id):
        raise HTTPException(status_code=404, detail="Product not found")
    q = text("SELECT SUM(manufacturing_hours) AS total FROM productionprocess WHERE product_id = :pid")
    try:
        with engine.connect() as conn:
            res = conn.execute(q, {"pid": product_id}).fetchone()
            total = res[0]
            if total is None:
                return {"product_id": product_id, "manufacture_time_hours": 0}
            return {"product_id": product_id, "manufacture_time_hours": int(math.ceil(total))}
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/calc-raw")
def calc_raw(product_type_id: int, material_type_id: int, quantity: int, p1: float, p2: float):
    if quantity <= 0 or p1 <= 0 or p2 <= 0:
        return {"result": -1}
    try:
        with engine.connect() as conn:
            coef_row = conn.execute(select(producttype.c.coefficient).where(producttype.c.product_type_id == product_type_id)).fetchone()
            if coef_row is None:
                return {"result": -1}
            coef = float(coef_row[0])
            loss_row = conn.execute(select(materialtype.c.loss_percentage).where(materialtype.c.material_type_id == material_type_id)).fetchone()
            if loss_row is None:
                return {"result": -1}
            loss = float(loss_row[0])
    except SQLAlchemyError:
        return {"result": -1}
    per_unit = p1 * p2 * coef
    total = per_unit * quantity
    total_with_loss = total * (1 + loss)
    return {"result": int(math.ceil(total_with_loss))}
