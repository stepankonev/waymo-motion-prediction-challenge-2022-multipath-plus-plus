import multiprocessing
from tqdm import tqdm
import tensorflow as tf
from utils.prerender_utils import get_visualizers, create_dataset, parse_arguments, merge_and_save
from utils.utils import get_config
from utils.features_description import generate_features_description

def main():
    args = parse_arguments()
    dataset = create_dataset(args.data_path, args.n_shards, args.shard_id)
    visualizers_config = get_config(args.config)
    visualizers = get_visualizers(visualizers_config)

    p = multiprocessing.Pool(args.n_jobs)
    processes = []
    k = 0
    for data in tqdm(dataset.as_numpy_iterator()):
        k += 1
        data = tf.io.parse_single_example(data, generate_features_description())
        processes.append(
            p.apply_async(
                merge_and_save,
                kwds=dict(
                    visualizers=visualizers,
                    data=data,
                    output_path=args.output_path,
                ),
            )
        )

    for r in tqdm(processes):
        r.get()

if __name__ == "__main__":
    main()