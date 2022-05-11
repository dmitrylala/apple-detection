from utils.generate import get_strides, get_n_patches, get_actual_overlap


def main():
    data = [
        {
            'img_shape': (3000, 4000),
            'overlapping': (0.3, 0.9),
            'kernel_size': (1024, 1024)
        },
        {
            'img_shape': (3000, 2700),
            'overlapping': (0.3, 0.5),
            'kernel_size': (1024, 1024)
        }
    ]
    for in_data in data:
        strides = get_strides(**in_data)
        n_patches = get_n_patches(in_data['img_shape'], strides, in_data['kernel_size'])
        overlap = get_actual_overlap(strides, in_data['kernel_size'])
        print(f"{strides=}, {n_patches=}, {overlap=}")


if __name__ == '__main__':
    main()
