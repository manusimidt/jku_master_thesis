# This work is still in progress!


# Replication
The following section will contain all commands used to run the different experiments for easy replication. Please note 
that for CoinRun and crafter additional demonstration datasets are needed which are NOT included
in this repository. The jumping task experiments can be run without an additional dataset.

All experiments where performed on a system running on an intel core i5-13600K and an Nvidia GTX 1660 SUPER with 
32GB RAM and 6 GB VRAM.
The jumping task experiments took 12-20 minutes on average per run. The crafter experiments took 8-14 hours 
and the coinrun experiments took 2-5 hours. 

## jumping task commands:
```bash
python -m jumping.train -a1 0 -a2 1 -s 10 20 30 -aug identity --conf wide_grid -K 5000
python -m jumping.train -a1 0 -a2 1 -s 10 20 30 -aug identity --conf narrow_grid -K 5000
python -m jumping.train -a1 0 -a2 1 -s 10 20 30 -aug identity --conf random_grid -K 5000

python -m jumping.train -a1 0 -a2 1 -s 10 20 30 -aug blur_noise --conf wide_grid -K 5000
python -m jumping.train -a1 0 -a2 1 -s 10 20 30 -aug blur_noise --conf narrow_grid -K 5000
python -m jumping.train -a1 0 -a2 1 -s 10 20 30 -aug blur_noise --conf random_grid -K 5000

python -m jumping.train -a1 0 -a2 1 -s 10 20 30 -aug conv --conf wide_grid -K 8000
python -m jumping.train -a1 0 -a2 1 -s 10 20 30 -aug conv --conf narrow_grid -K 8000
python -m jumping.train -a1 0 -a2 1 -s 10 20 30 -aug conv --conf random_grid -K 8000

python -m jumping.train -a1 0 -a2 1 -s 10 20 30 -aug crop2 --conf wide_grid -K 8000
python -m jumping.train -a1 0 -a2 1 -s 10 20 30 -aug crop2 --conf narrow_grid -K 8000
python -m jumping.train -a1 0 -a2 1 -s 10 20 30 -aug crop2 --conf random_grid -K 8000

python -m jumping.train -a1 5 -a2 1 -s 10 20 30 -aug identity --conf wide_grid   -psm f -K 10000
python -m jumping.train -a1 5 -a2 1 -s 10 20 30 -aug identity --conf narrow_grid -psm f -K 10000
python -m jumping.train -a1 5 -a2 1 -s 10 20 30 -aug identity --conf random_grid -psm f -K 10000

python -m jumping.train -a1 5 -a2 1 -s 10 20 30 -aug conv --conf wide_grid   -psm f -K 10000
python -m jumping.train -a1 5 -a2 1 -s 10 20 30 -aug conv --conf narrow_grid -psm f -K 10000
python -m jumping.train -a1 5 -a2 1 -s 10 20 30 -aug conv --conf random_grid -psm f -K 10000

python -m jumping.train -a1 5 -a2 1 -s 10 20 30 -aug crop2 --conf wide_grid   -psm f -K 10000
python -m jumping.train -a1 5 -a2 1 -s 10 20 30 -aug crop2 --conf narrow_grid -psm f -K 10000
python -m jumping.train -a1 5 -a2 1 -s 10 20 30 -aug crop2 --conf random_grid -psm f -K 10000

python -m jumping.train -a1 5 -a2 1 -s 10 20 30 -aug identity --conf wide_grid   -psm fb -K 10000
python -m jumping.train -a1 5 -a2 1 -s 10 20 30 -aug identity --conf narrow_grid -psm fb -K 10000
python -m jumping.train -a1 5 -a2 1 -s 10 20 30 -aug identity --conf random_grid -psm fb -K 10000

python -m jumping.train -a1 5 -a2 1 -s 10 20 30 -aug conv --conf wide_grid   -psm fb -K 10000
python -m jumping.train -a1 5 -a2 1 -s 10 20 30 -aug conv --conf narrow_grid -psm fb -K 10000
python -m jumping.train -a1 5 -a2 1 -s 10 20 30 -aug conv --conf random_grid -psm fb -K 10000

python -m jumping.train -a1 5 -a2 1 -s 10 20 30 -aug crop2 --conf wide_grid   -psm fb -K 10000
python -m jumping.train -a1 5 -a2 1 -s 10 20 30 -aug crop2 --conf narrow_grid -psm fb -K 10000
python -m jumping.train -a1 5 -a2 1 -s 10 20 30 -aug crop2 --conf random_grid -psm fb -K 10000

```

## crafter commands
```bash
python -m crafter.train -a1 0 -a2 1 -s 10 20 30 --K 500000 
python -m crafter.train -a1 2 -a2 1 -s 10 20 30 --K 500000 --psm f
python -m crafter.train -a1 2 -a2 1 -s 10 20 30 --K 500000 --psm fb
```

## coinrun commands
```bash
python -m coinrun.train -a1 0 -a2 1 -s 10 20 30 --aug identity
python -m coinrun.train -a1 0 -a2 1 -s 10 20 30 --aug conv
python -m coinrun.train -a1 0 -a2 1 -s 10 20 30 --aug noise

python -m coinrun.train -a1 5 -a2 1 -s 10 20 30 --aug identity --psm f
python -m coinrun.train -a1 5 -a2 1 -s 10 20 30 --aug conv --psm f
python -m coinrun.train -a1 5 -a2 1 -s 10 20 30 --aug noise --psm f

python -m coinrun.train -a1 5 -a2 1 -s 10 20 30 --aug identity --psm fb
python -m coinrun.train -a1 5 -a2 1 -s 10 20 30 --aug conv --psm fb
python -m coinrun.train -a1 5 -a2 1 -s 10 20 30 --aug noise --psm fb
```