import logging


def evaluate(input_data, image_path):
    clip_min_row = input_data.loc[input_data['clip1'].idxmin()]
    clip_min_id = clip_min_row['id']
    clip1_min_value = clip_min_row['clip1']
    clip_min_name = clip_min_row['name']
    clip_min_des = clip_min_row['description']

    logging.info("Clip min: %s", clip1_min_value)
    logging.info("Clip min id: %s", clip_min_id)

    if clip1_min_value < 0.5:
        return {'name': clip_min_name, 'description': clip_min_des}
    else:
        return None
