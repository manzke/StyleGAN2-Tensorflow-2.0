if __name__ == '__main__':
  import argparse


  parser = argparse.ArgumentParser()

  parser.add_argument('--datapath', type=str, required=True)
  parser.add_argument('--dataset', type=str, required=True)
  parser.add_argument('--resultspath', type=str, required=True)
  parser.add_argument('--modelspath', type=str, required=True)

  parser.add_argument('--maxsteps', type=int, default=1000000)
  parser.add_argument('--lr', type=float, default=0.0001)

  parser.add_argument('--img_size', type=int, default=256)

  parser.add_argument('--verbose', action='store_true')

  args = parser.parse_args()

  from stylegan_two import StyleGAN

  model = StyleGAN(lr = args.lr, verbose=args.verbose,
                   max_steps = args.maxsteps,
                   data_dir=args.datapath, dataset=args.dataset,
                   results_dir=args.resultspath, models_dir=args.modelspath,
                   im_size=args.img_size)

  model.evaluate(0)

  while model.GAN.steps <=model.max_steps:
    print(model.GAN.steps, model.max_steps)
    model.train()
