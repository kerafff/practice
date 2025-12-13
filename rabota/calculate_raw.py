def calc_raw(
    product_type_id: int,
    material_type_id: int,
    quantity: int,
    p1: float,
    p2: float,
    product_coeffs: dict,
    material_losses: dict
) -> int:
    """
    Расчёт количества сырья для производства продукции.

    :param product_type_id: идентификатор типа продукции (int)
    :param material_type_id: идентификатор типа материала (int)
    :param quantity: количество продукции (int)
    :param p1: параметр продукции 1 (float, > 0)
    :param p2: параметр продукции 2 (float, > 0)
    :param product_coeffs: коэффициенты типов продукции {id: coeff}
    :param material_losses: потери сырья по типу материала {id: loss_percent}
    :return: целое количество сырья или -1 при ошибке
    """

    # Проверка входных данных
    if (
        product_type_id not in product_coeffs or
        material_type_id not in material_losses or
        quantity <= 0 or
        p1 <= 0 or
        p2 <= 0
    ):
        return -1

    try:
        product_coeff = float(product_coeffs[product_type_id])
        loss_percent = float(material_losses[material_type_id])

        # Расход сырья на одну единицу продукции
        raw_per_unit = p1 * p2 * product_coeff

        # Общий расход без учёта потерь
        total_raw = raw_per_unit * quantity

        # Увеличение с учётом потерь сырья
        total_with_losses = total_raw * (1 + loss_percent / 100)

        return int(round(total_with_losses))
    except Exception:
        return -1
