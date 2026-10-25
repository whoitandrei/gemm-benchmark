# gemm-benchmark

## Структура

```
src/
  problem.hpp/.cpp     задача GEMM: размеры, пакет, тип данных, раскладка; генерация входов
  params.hpp/.cpp      параметры ядра и перебор их значений
  reference.hpp/.cpp   эталон (double) и проверка точности результата
  kernel.hpp           интерфейс ядра: prepare() -> run() -> readC()
  kernels/
    registry.cpp       список всех ядер сборки
    cpu_tiled.cpp      CPU-ядро
bench/
  main.cpp             запуск Catch2 + опции перебора 
  config.hpp/.cpp      разбор опций в список задач
  bench_gemm.cpp       перебор ядро x задача x параметры, BENCHMARK_ADVANCED
  results.hpp/.cpp     listener Catch2: метрики GOP/s, GB/s, сводная таблица, CSV
tests/
  test_gemm.cpp        корректность всех ядер (Catch2)
scripts/
  plot.py              график GOP/s по CSV
```

## Сборка


```bash
cmake -S . -B build -G Ninja
```

```bash
cmake --build build
```

## Запуск

Тесты корректности:

```bash
ctest --test-dir build --output-on-failure
```

Бенчмарк с настройками по умолчанию (размеры 128, 256, 512; f32 и i8; все параметры ядер):

```bash
./build/gemm_bench --csv results.csv
```

В конце печатается сводная таблица:

```
kernel     params              problem                runs       mean      stddev    GOP/s  GB/s  max err
cpu_tiled  BK=64 BM=16 BN=256  f32 256x256x256 b1 nn  190+100x1  1.053 ms  18.77 us  31.85  0.75  1.5e-07
...
```

Примеры перебора:

```bash
./build/gemm_bench --shapes 1024x1024x64,4096 --dtype f32 --params "BM=16,32,64,128;BN=256"
```

```bash
./build/gemm_bench --shapes 64 --batch 1,16,256 --broadcast-b --trans nn,nt
```

График:

```bash
python3 scripts/plot.py results.csv -o results.png
```

```bash
python3 scripts/plot.py results.csv --best -o best.png
```

Только проверить корректность всех комбинаций, без замеров:

```bash
./build/gemm_bench --skip-benchmarks --verify-samples 0
```

### Опции перебора

| Опция | Что задаёт | По умолчанию |
|---|---|---|
| `--kernel a,b` | ядра | все |
| `--shapes 256,1024x512x64` | размеры: `N` (квадратная) или `MxNxK` | `128,256,512` |
| `--batch 1,8` | размер пакета | `1` |
| `--broadcast-b` | одна матрица B на весь пакет | выкл. |
| `--dtype f32,i8` | типы данных | `f32,i8` |
| `--trans nn,nt,tn,tt` | транспонирование A и B | `nn` |
| `--params "BM=32,64;BK=128"` | значения параметров ядра вместо значений по умолчанию | из ядра |
| `--csv file` | CSV со всеми метриками | нет |
| `--verify-samples n` | сколько элементов C сверять с эталоном, `0` = все | `1024` |
| `--warmup-runs n` | минимум холостых запусков ядра перед замером | `3` |
| `--warmup-ms t` | минимум времени прогрева этими запусками, мс | `200` |
| `--list-kernels` | ядра и их параметры | |

Опции Catch2 для замеров: `--benchmark-samples` (по умолчанию 100), `--benchmark-confidence-interval`,
`--benchmark-no-analysis`, `--skip-benchmarks`. `--benchmark-warmup-time` здесь по умолчанию 0:
прогрев Catch2 только опрашивает часы и не запускает ядро. Вместо него — `--warmup-runs` и `--warmup-ms`.
Все опции: `./build/gemm_bench --help`.

### Параметры замера

| Параметр | Опция | Статус |
|---|---|---|
| Число выборок | `--benchmark-samples` (100) | есть |
| Прогрев | `--warmup-runs` (3), `--warmup-ms` (200) | есть |
| Проверка точности | `--verify-samples` (1024, 0 = все элементы) | есть |
| Холодный кэш | `--cold-cache`: сброс кэша между запусками | план |
| Время на GPU | timestamp queries вместо времени на CPU | план |
| Устройство | `--device`, если GPU несколько | план |

## Что замеряется

Метрики:

- `GOP/s = 2·M·N·K·batch / mean`;
- `GB/s = (байты A + B + C) / mean`;
- `max_error = max |c − ref| / Σ|a·b|`. Для f32 допуск `(K + 2)·2⁻²⁴`, для i8 нужно точное совпадение.

## Как добавить ядро

1. Создать `src/kernels/<name>.cpp`: наследники `GemmKernel` и `GemmInstance`
   и фабрика `std::unique_ptr<GemmKernel> make<Name>Kernel()`.
2. В `paramSpace()` вернуть параметры и значения для перебора.
   В `supports()` отсечь неподдерживаемые типы, раскладки и параметры.
3. Добавить фабрику в `src/kernels/registry.cpp` и файл в `gemm_core` в `CMakeLists.txt`.

