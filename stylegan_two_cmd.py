if __name__ == '__main__':
  import argparse


  parser = argparse.ArgumentParser()

  parser.add_argument('--datadir', type=str, required=True)
  parser.add_argument('--dataset', type=str, required=True)
  parser.add_argument('--resultsdir', type=str, required=True)
  parser.add_argument('--modelsdir', type=str, required=True)

  parser.add_argument('--maxsteps', type=int, default=1000000)
  parser.add_argument('--lr', type=float, default=0.0001)

  parser.add_argument('--im_size', type=int, default=256)

  parser.add_argument('--verbose', action='store_true')

  args = parser.parse_args()

  from stylegan_two import StyleGAN

  model = StyleGAN(lr = args.lr, verbose=args.verbose,
                   max_steps = args.maxsteps,
                   data_dir=args.datadir, dataset=args.dataset,
                   results_dir=args.resultsdir, models_dir=args.modelsdir,
                   im_size=args.im_size)

  model.evaluate(0)

  while model.GAN.steps <=model.max_steps:
    print(model.GAN.steps, model.max_steps)
    model.train()
