# gan_edu

1. рисуем архитектуру: ✓
      - на основе DCGAN ✓
      - блоки меняем на csp ✓
2. запустить обучение на celeba ✓
3. логировать лоссы и промежуточные картинки в clearml или tensorboard\weightandbias ✓
4. добиться сходимости +-
5. попробовать разные варианты лоссов и регуляризаций ✓
6. сравнить обучение при разных подходах ✓
7. сравнить с сеткой на Resnet блоках классических ✓

### Получилось, что скрипты по большей части схожи, но решил не городить опциональность в одном скрипте (тип блоков, регуляризации и т.д.) в угоду наглядности по папкам.

# Classic conv2d DCGAN

#### ClearML:

#### Из опробованного:
batchnorm, instancenorm, dropout <br>
разные batchsize, LR, adamW <br>

#### остановился на конфиге:
- batchnorm
- dropout p=0.1 перед skip-connection слоями
- BS 32
- adam для G lr 1e-4
- adamw для D lr 2e-5

# DCGAN CSP

#### ClearML:

#### Из опробованного:
batchnorm, instancenorm, dropout <br>
разные batchsize, LR, adamW <br>

#### остановился на конфиге:
- batchnorm
- dropout p=0.1 перед skip-connection слоями
- BS 32
- adam для G lr 1e-4
- adamw для D lr 2e-5

# DCGAN ResNet
#### ClearML:

#### Из опробованного:
batchnorm, instancenorm, dropout <br>
разные batchsize, LR, adamW <br>

#### остановился на конфиге:
- batchnorm
- dropout p=0.1 перед skip-connection слоями
- BS 32
- adam для G lr 1e-4
- adamw для D lr 2e-5