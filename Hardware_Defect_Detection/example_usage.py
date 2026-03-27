"""
使用示例 - 演示完整的数据工程流程
"""
from hardware_defect_system import DefectDataEngine, Config, create_sample_data


def example_basic_usage():
    """基础使用示例"""
    print("=" * 60)
    print("硬件缺陷检测数据工程系统 - 基础示例")
    print("=" * 60)

    config = Config()

    csv_path, image_dir = create_sample_data(
        output_dir='./sample_data',
        num_samples=100
    )

    engine = DefectDataEngine(config.to_dict())

    results = engine.run_full_pipeline(
        input_csv=csv_path,
        image_dir=image_dir,
        output_format='coco'
    )

    print("\n处理结果统计:")
    for key, value in results.items():
        print(f"{key}: {value}")


def example_with_training():
    """包含模型训练的完整示例"""
    print("=" * 60)
    print("硬件缺陷检测数据工程系统 - 完整训练示例")
    print("=" * 60)

    config = Config()
    config.NUM_EPOCHS = 5
    config.BATCH_SIZE = 8

    csv_path, image_dir = create_sample_data(
        output_dir='./training_data',
        num_samples=500
    )

    engine = DefectDataEngine(config.to_dict())

    engine.run_full_pipeline(
        input_csv=csv_path,
        image_dir=image_dir,
        output_format='coco'
    )

    engine.train_model(
        train_csv=csv_path,
        image_dir=image_dir,
        model_type='fasterrcnn',
        epochs=config.NUM_EPOCHS,
        lr=config.LEARNING_RATE
    )


def example_custom_pipeline():
    """自定义 Pipeline 示例"""
    from feature_extractor import DefectFeatureExtractor
    from outlier_filter import OutlierFilter
    from annotation_converter import AnnotationConverter
    from data_pipeline import DataPipeline
    import pandas as pd

    print("=" * 60)
    print("自定义数据处理 Pipeline")
    print("=" * 60)

    csv_path = './data/annotations.csv'
    image_dir = './data/images'

    df = pd.read_csv(csv_path)

    extractor = DefectFeatureExtractor()
    features_df = extractor.batch_extract(df)
    features_df.to_csv('./output/features.csv', index=False)

    filter = OutlierFilter(contamination=0.05)
    filter.fit(features_df)
    filtered_df = filter.filter_outliers(features_df)
    filtered_df.to_csv('./output/filtered_features.csv', index=False)

    classes = {'defect': 1, 'scratch': 2, 'dent': 3}
    converter = AnnotationConverter(classes)
    converter.csv_to_coco(csv_path, './output/coco.json', image_dir)

    pipeline = DataPipeline(img_size=512, batch_size=16)
    train_loader = pipeline.create_train_loader(csv_path, image_dir)

    print(f"训练批次数量：{len(train_loader)}")


if __name__ == '__main__':
    example_basic_usage()
