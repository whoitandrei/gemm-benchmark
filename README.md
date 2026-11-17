# gemm-benchmark

Внешний бенчмарк GEMM: замеры чужих библиотек, с которыми сравниваются Vulkan-ядра Adept. Сами Vulkan-ядра
Adept меряются в Adept, программой `benchmarks/op_bench`: так они работают через настоящий пайплайн библиотеки.
Этот бенчмарк меряет ориентиры тем же способом и пишет CSV в том же формате, поэтому результаты обеих программ
можно сложить в одну таблицу и один график. План работ — [docs/roadmap.md](docs/roadmap.md).

## Структура

```
src/
  device.hpp/.cpp      название процессора и число потоков
  problem.hpp/.cpp     задача GEMM: пакет, M, K, N, типы входа и накопления, общая B; генерация входов
  half.hpp             f16: перевод float <-> half с округлением к ближайшему чётному
  reference.hpp/.cpp   эталон (double) и проверка точности результата
  kernel.hpp           интерфейс ядра: prepare() -> run() ... sync() -> readC(); таймер GPU на блок вызовов
  measure.hpp/.cpp     замер как в Adept (benchmark::measure): прогрев, раунды по N вызовов, медианы
  kernels/
    registry.cpp       список всех ядер сборки
    ggml_cpu.cpp       ggml, CPU-бэкенд (ggml_mul_mat)
    xnnpack.cpp        XNNPACK, fully connected f32
    ggml_vulkan.cpp    ggml, Vulkan-бэкенд (ggml_mul_mat): f32, или f16 и тензорные ядра при GEMM_GGML_VULKAN_F16=1
    cublas.cpp         cuBLAS (cublasGemmStridedBatchedEx), только NVIDIA: ядра cublas, cublas_tc, cublas_tf32
bench/
  main.cpp             gemm_bench: опции как у op_bench, проверка точности, замер, отчёт
  cases.hpp/.cpp       чтение датасета случаев (CSV)
  report.hpp/.cpp      строки RESULT и CSV в формате op_bench
datasets/
  matmul.csv           основной датасет: случаи GEMM f32, общий с op_bench Adept (--matmul-cases)
  matmul_f16.csv       синтетические формы в f16/f32 и f16/f16 — теоретическое сравнение, Adept их не считает
tests/
  test_gemm.cpp        эталон, f16, ядра после вызовов в очереди, gpu_ms, замер (Catch2)
scripts/
  plot.py              GOP/s по CSV gemm_bench и op_bench
  peaks.csv            пики устройств с источниками, описание в docs/devices.md
  gpu_counters.py      счётчики GPU M1 из трассы Instruments, по каждому compute pipeline
```

## Сборка

```bash
cmake -S . -B build -G Ninja
```

```bash
cmake --build build
```

Ядра подключаются опциями CMake. По умолчанию собирается только `ggml_cpu` (`GEMM_WITH_GGML=ON`) и `cublas`,
если CMake нашёл CUDA Toolkit. Если не включено ни одно ядро, CMake останавливается с ошибкой `no kernels`.

| Опция | Ядра | По умолчанию |
|---|---|---|
| `GEMM_WITH_GGML` | `ggml_cpu` | `ON` |
| `GEMM_WITH_XNNPACK` | `xnnpack` | `OFF`, сборка несколько минут |
| `GEMM_WITH_GGML_VULKAN` | `ggml_vulkan` | `OFF`, нужен Vulkan SDK, сборка долгая |
| `GEMM_WITH_CUBLAS` | `cublas`, `cublas_tc`, `cublas_tf32` | `ON`, если найден CUDA Toolkit 11.4+ |

Все библиотеки, кроме CUDA, скачиваются при первой настройке CMake.

### ggml и XNNPACK

```bash
cmake -S . -B build -DGEMM_WITH_XNNPACK=ON
```

- `ggml_cpu` — ggml v0.26.0, только CPU-бэкенд, умножение через `ggml_mul_mat` с llamafile sgemm,
  как в llama.cpp. Свой пул потоков, без OpenMP.
- `xnnpack` — fully connected оператор f32. Пакет с общей B считается одним оператором,
  иначе — отдельный оператор на каждую матрицу пакета.
- Обе считают B весами: упаковка B (у ggml — переупаковка в N x K) идёт в `prepare()` и не замеряется,
  как в инференсе.
- Только f32, пакет и общая B поддерживаются.
- Потоков столько, сколько ядер у машины (`cpuThreads()`, то есть `std::thread::hardware_concurrency`). На процессорах с
  энергоэффективными ядрами (Apple M, Intel 12+) это не всегда быстрее: ggml делит работу между потоками
  поровну, и потоки на медленных ядрах тормозят весь запуск.

### ggml-vulkan

Vulkan-бэкенд той же ggml v0.26.0, что у `ggml_cpu`, — GEMM из llama.cpp. Зрелая открытая реализация на
том же API, что Adept. На M1 это единственный внешний GPU-ориентир.

```bash
cmake -S . -B build -DGEMM_WITH_GGML_VULKAN=ON
```

- Нужен Vulkan SDK с `glslc` и SPIRV-Headers. На macOS SDK работает через MoltenVK; если SDK поставлен не
  системно, перед cmake и запуском: `source ~/VulkanSDK/<версия>/setup-env.sh`. ggml скачивается, сборка
  долгая: ggml генерирует и компилирует несколько сотен шейдеров.
- Умножение — `ggml_mul_mat`, один граф на весь пакет, общая B — broadcast ggml. B считается весами, как
  у `ggml_cpu`: перестановка в N x K идёт в `prepare()`. Транспонированная A не поддерживается.
- Два режима, режим выбирается на весь процесс:

  | Режим | Как включить | Имя ядра | Типы | Что делает ядро |
  |---|---|---|---|---|
  | f32 (по умолчанию) | — | `ggml_vulkan` | f32/f32 | до запуска ggml выставляет `GGML_VK_DISABLE_F16`, `GGML_VK_DISABLE_COOPMAT`, `GGML_VK_DISABLE_COOPMAT2` |
  | f16 | `GEMM_GGML_VULKAN_F16=1` | `ggml_vulkan_tc`, если ggml включил тензорные ядра, иначе `ggml_vulkan_f16` | f32/f32, f16/f32, f16/f16 | переменные ggml не трогает; f32-входы ggml считает через f16 |

  Включил ли ggml тензорные ядра, ядро узнаёт из строки ggml `matrix cores:` (`KHR_coopmat`, `NV_coopmat2` или
  `none`): ggml печатает её при старте, ядро перехватывает её через `ggml_log_set` и передаёт в stderr дальше.
  Тот же режим видно в строке `device:`.

  Зачем режим f32 выключает f16 и тензорные ядра. На GPU с f16 ggml собирает матричный шейдер и для
  f32-входов с `FLOAT_TYPE = float16_t` (`vulkan-shaders-gen.cpp`): тайлы A и B в shared memory
  (`mul_mm.comp`, `buf_a`, `buf_b`) хранятся в f16, f32-входы округляются. Шейдеры cooperative matrix собраны
  так же и выбираются по поддержке coopmat, а не по флагу f16 (`ggml-vulkan.cpp`, `matmul_f32_f32` через
  `cm1_create`). Проверка на M1 при `--init const`: в режиме f16 ошибка на `odd_127` и `odd_1000` — 4.3e-4, это
  округление 0.01 до f16; строгий допуск f32 (7.7e-6 и 6.0e-5) такой результат не пропускает. Поэтому в режиме
  f16 у f32-входов допуск шире — раздел «Типы данных».
- f16: `f16/f32` — `GGML_PREC_F32`, `f16/f16` — `GGML_PREC_DEFAULT`, ggml копит в f16 (`ACC_TYPE float16_t`).
  Результат `ggml_mul_mat` всегда f32 (`ggml.c`), для накопления в f16 он переводится в f16 в `readC()`, вне
  замера.
- Вызов — `ggml_backend_graph_compute_async`, ожидание — `ggml_backend_synchronize`, как у Adept: вызовы
  блока идут в очередь, ожидание одно на блок. ggml записывает command buffer заново при каждом вызове, эта
  работа CPU входит в `cpu_ms` и `wall_ms`. Если устройство не поддерживает асинхронный режим, ggml сам ждёт
  конца каждого вызова.
- Устройство: первая дискретная GPU из списка ggml, потом встроенная. Выбрать вручную —
  `GGML_VK_VISIBLE_DEVICES=<номер>`. Список устройств и их возможности (`fp16`, `matrix cores`) ggml
  печатает в stderr при старте.
- На NVIDIA в режиме f16 ggml берёт тензорные ядра через cooperative matrix. Без них:
  `GEMM_GGML_VULKAN_F16=1 GGML_VK_DISABLE_COOPMAT=1 GGML_VK_DISABLE_COOPMAT2=1`, ядро тогда называется
  `ggml_vulkan_f16`. В режиме f32 они выключены всегда.
- `gpu_ms` нет (`nan`): ggml не отдаёт timestamp queries наружу. Время каждой операции на GPU ggml печатает
  сам при `GGML_VK_PERF_LOGGER=1`.

### cuBLAS

Нужен CUDA Toolkit 11.4 или новее. Если CMake находит его, ядра собираются сами, если нет — проект
собирается без них. Явно: `-DGEMM_WITH_CUBLAS=ON` (без toolkit — ошибка CMake) или `OFF`.
Toolkit не на стандартном месте:

```bash
cmake -S . -B build -DCUDAToolkit_ROOT=/usr/local/cuda
```

- Считает первая видимая GPU. Выбрать другую: `CUDA_VISIBLE_DEVICES=<номер>` (номера показывает
  `nvidia-smi -L`). В stderr: `cuda: <имя GPU> (sm_89)`. Нет устройства — ядра пропускаются, причина в stderr.
- Весь пакет — один вызов `cublasGemmStridedBatchedEx`, алгоритм `CUBLAS_GEMM_DEFAULT`. cuBLAS хранит
  матрицы по столбцам, поэтому ядро считает Cᵀ = Bᵀ·Aᵀ: A и B меняются местами в вызове, без копий.
  Общая B — шаг 0 между матрицами пакета.
- Вызов асинхронный, ожидание — `cudaDeviceSynchronize`. `gpu_ms` — CUDA events в начале и в конце блока
  вызовов, как timestamp queries у Adept. Буферы выделяются и заполняются в `prepare()`.

Три ядра фиксируют тип вычислений (compute type), чтобы cuBLAS не выбирал его сам:

| Вход / накопление | `cublas` | `cublas_tc` | `cublas_tf32` |
|---|---|---|---|
| f32 / f32 | `CUBLAS_COMPUTE_32F_PEDANTIC` | `CUBLAS_COMPUTE_32F_FAST_16F` | `CUBLAS_COMPUTE_32F_FAST_TF32` |
| f16 / f32 | `CUBLAS_COMPUTE_32F_PEDANTIC` | `CUBLAS_COMPUTE_32F` | нет |
| f16 / f16 | `CUBLAS_COMPUTE_16F_PEDANTIC` | `CUBLAS_COMPUTE_16F` | нет |
| GPU | любая; f16/f16 — sm_53+ | sm_70+ | sm_80+ |

- `cublas` — pedantic: все шаги строго в типе накопления. Это основное сравнение в f32. Для `16F_PEDANTIC`
  документация cuBLAS прямо говорит, что тензорные ядра выключены, для `32F_PEDANTIC` — нет. Какие CUDA-ядра
  запустились, видно в Nsight Systems: `nsys profile --stats=true ./build/gemm_bench ...`.
- `cublas_tc` — тензорные ядра f16. f16-входы идут как есть. f32-входы cuBLAS сам переводит в f16
  (`FAST_16F`), накопление и C — f32. Так же может работать ядро Vulkan на cooperative matrix, если f32 из памяти
  переводить в f16 по дороге в shared memory.
- `cublas_tf32` — тензорные ядра TF32 для f32-входов. В Vulkan TF32 нет, это потолок CUDA, а не цель для Adept.
- Ядра на GPU слабее указанной в таблице не запускаются, чтобы имя `_tc` не стояло у замера без тензорных ядер.

Частоты GPU плавают из-за boost и нагрева, поэтому перед замерами на NVIDIA их фиксируют. Это действует
и на Vulkan, в том числе на `op_bench` Adept. Поддерживаемые частоты:

```bash
nvidia-smi --query-supported-clocks=graphics,memory --format=csv
```

Зафиксировать частоту ядра GPU (на Linux заодно включить persistence mode, на Windows — без `sudo`
из консоли администратора):

```bash
sudo nvidia-smi --persistence-mode=1
```

```bash
sudo nvidia-smi --lock-gpu-clocks=<МГц>,<МГц>
```

Частоту брать такую, которую GPU держит под нагрузкой: во время замера `clocks.sm` должна стоять на ней.

```bash
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv -l 1
```

Частоту памяти фиксирует `--lock-memory-clocks=<МГц>,<МГц>` (поддерживают не все GPU). После замеров:

```bash
sudo nvidia-smi --reset-gpu-clocks
```

## Запуск

Тесты корректности и короткий прогон `gemm_bench`. При `GEMM_WITH_GGML_VULKAN` тесты идут дважды: второй раз —
`gemm_tests_ggml_vulkan_f16`, `ggml_vulkan` в режиме f16:

```bash
ctest --test-dir build --output-on-failure
```

Все ядра на всех случаях датасета по умолчанию (`datasets/matmul.csv`), отчёт в `results/gemm_bench_latest.csv`:

```bash
./build/gemm_bench
```

Вывод — как у `op_bench`: строка `device:` на каждое ядро, строка `init:` с режимом заполнения матриц, затем
строка `RESULT` на каждый замер.

```
device: ggml_cpu cpu Apple M1 Pro
device: ggml_vulkan vulkan Apple M1 Pro, f32 mode: fp16 and coopmat off
RESULT case=square_2048 phase=matmul shape=1x2048x2048x2048 wall_ms=30.0378521 gpu_ms=nan cpu_ms=30.0378521 kernel=ggml_cpu
RESULT case=square_2048 phase=matmul shape=1x2048x2048x2048 wall_ms=32.7694167 gpu_ms=nan cpu_ms=0.0138646 kernel=ggml_vulkan
report: results/gemm_bench_latest.csv
```

Только GPU-ядра на квадратных формах, свой файл отчёта:

```bash
./build/gemm_bench --device gpu --case square --report results/square_gpu.csv
```

| Опция | Что задаёт | По умолчанию |
|---|---|---|
| `--matmul-cases file.csv` | датасет случаев, формат — раздел «Датасет случаев» | `datasets/matmul.csv` |
| `--device cpu\|gpu` | только CPU-ядра или только GPU-ядра | все |
| `--kernel a,b` | ядра по имени | все |
| `--iters n` | вызовов в замеряемом блоке | `20` |
| `--warmup n` | холостых вызовов перед замером | `5` |
| `--rounds n` | замеряемых блоков; в отчёт идут медианы | `5` |
| `--case s` | группы случаев: часть имени до первого `_` содержит `s` (`square`, `inference`, `batch`, `odd`, `matmul`, `da3s`, `da3l`, `dec`, `pre`, `moe`, `attn`, `vit`, `asr`) | все |
| `--case-name s` | случаи, в имени которых есть `s` | все |
| `--report path` | файл CSV | `results/gemm_bench_latest.csv` |
| `--init const\|normal\|uniform:LOW,HIGH` | чем заполнить A и B — раздел «Заполнение матриц» | `const` |
| `--list` | ядра и случаи | |

Как у `op_bench`, при любом фильтре (`--device`, `--kernel`, `--case`, `--case-name`) отчёт пишется, только
если задан `--report`: частичный прогон не затирает полный. Код возврата 1 — ядро не прошло проверку точности,
упало или не замерено ни одного случая.

### Сравнение с Adept

Vulkan-ядра Adept меряет `op_bench` из репозитория Adept (`benchmarks/`, собирается с `WITH_BENCHMARKS`).
Обе программы читают один и тот же датасет через `--matmul-cases`, поэтому меряют одни и те же случаи под
одними именами. Сравнение с Adept идёт в f32: в Adept нет типа f16 (`dtype_t` в `include/adept/types.hpp`, GEMM —
`DISPATCH_FLOATING_TYPES`, только Float32 и Float64). Основной датасет `datasets/matmul.csv` — только f32/f32. Если в
файле есть другие строки, `op_bench` их пропускает и пишет, сколько пропустил. Без `--matmul-cases` `op_bench` меряет свой встроенный `matmul_attention`, как раньше.
Опция `--matmul-cases` в `op_bench` — наша доработка Adept (`benchmarks/matmul_cases.*`), её нужно
согласовать с тимлидом.

Обе программы запускаются на одной машине, подряд, с одинаковыми параметрами замера и одинаковым `--init`:

```bash
<adept>/build-Release/benchmarks/op_bench --case matmul --matmul-cases <gemm-benchmark>/datasets/matmul.csv --init normal --warmup 10 --iters 20 --rounds 9 --report adept_normal.csv
```

```bash
./build/gemm_bench --kernel ggml_cpu,ggml_vulkan,cublas --init normal --warmup 10 --iters 20 --rounds 9 --report gemm_bench_normal.csv
```

Режим заполнения в CSV не пишется, поэтому его стоит указывать в имени файла отчёта.

### Заполнение матриц

Данные влияют на время: от них зависят потребление GPU и, через лимит мощности, его частота. Поэтому у обеих
программ одна и та же опция `--init` с одинаковыми режимами:

| Режим | Чем заполняются A и B |
|---|---|
| `const` | все элементы 0.01; по умолчанию, как `op_bench` до появления опции |
| `normal` | нормальное распределение со средним 0 и стандартным отклонением 0.02 — порядок весов трансформеров |
| `uniform:LOW,HIGH` | равномерное распределение в [LOW, HIGH), например `uniform:-1,1`; LOW < HIGH, обе границы конечны |

- Распределение в двух программах одно, а сами числа разные: генераторы разные. `gemm_bench` — `std::mt19937`
  с зерном 42; `op_bench` — `Tensor::full`, `Tensor::normal`, `Tensor::uniform` Adept.
- В `op_bench` режим действует на все операции, у которых входы создаёт `make_tensor`, а не только на GEMM.
- До 17.11 `gemm_bench` всегда заполнял матрицы равномерно в [−1, 1), а `op_bench` — константой 0.01.
  Замеры 13.11 соответствуют `--init uniform:-1,1` у `gemm_bench` и `--init const` у `op_bench`.
- При `const` все элементы C одинаковы, и проверка точности не заметит ошибку в индексах. Тесты (`ctest`)
  поэтому берут `uniform:-1,1`.
- В f16 числа округляются до f16. При широком диапазоне сумма может выйти за пределы f16, тогда ядро не пройдёт
  проверку.

f16 — только для оценки, насколько быстрее можно быть в теории, отдельным прогоном:

```bash
GEMM_GGML_VULKAN_F16=1 ./build/gemm_bench --matmul-cases datasets/matmul_f16.csv --kernel ggml_vulkan_tc,cublas,cublas_tc --warmup 10 --iters 20 --rounds 9 --report gemm_bench_f16.csv
```

### Тензорные ядра

Сколько дают тензорные ядра на f32-данных: те же случаи и данные, что у основного сравнения, но ядра считают
через f16 или TF32. Два прогона: режим ggml выбирается на весь процесс.

```bash
./build/gemm_bench --kernel cublas,cublas_tc,cublas_tf32 --init normal --warmup 10 --iters 20 --rounds 9 --report tc_cublas_normal.csv
```

```bash
GEMM_GGML_VULKAN_F16=1 ./build/gemm_bench --kernel ggml_vulkan_tc --init normal --warmup 10 --iters 20 --rounds 9 --report tc_ggml_normal.csv
```

| Ядро | Чем считает f32-данные | Аналог в Vulkan |
|---|---|---|
| `cublas` | CUDA-ядра, f32 | обычный шейдер Adept |
| `cublas_tc` | тензорные ядра: входы округляются до f16, накопление f32 | cooperative matrix f16 → f32 |
| `cublas_tf32` | тензорные ядра TF32, накопление f32 | нет: в Vulkan нет типа TF32 |
| `ggml_vulkan` | шейдер ggml, f32 | обычный шейдер |
| `ggml_vulkan_tc` | cooperative matrix ggml: тайлы f16 в shared memory, накопление f32 | он и есть |

Если строки `ggml_vulkan_tc` нет, а есть `ggml_vulkan_f16`, значит, ggml не включил тензорные ядра на этом
устройстве: строка `matrix cores:` в stderr.

Почему такие параметры: 9 раундов — нечётное число, медиана — реальный раунд, а не среднее двух; столько же
берёт `scripts/bench/run_op_regressions.py` Adept. 20 вызовов в раунде — по умолчанию `op_bench`, с ними
4096³ укладывается в разумное время. На M1 Pro основной датасет с `ggml_cpu` и `ggml_vulkan` — около 2.5 минуты,
f16-датасет с `ggml_vulkan` — меньше минуты.

График. Нужен `matplotlib`; его можно поставить в `.venv` проекта:

```bash
python3 -m venv .venv && .venv/bin/python -m pip install matplotlib
```

```bash
.venv/bin/python scripts/plot.py gemm_bench.csv adept.csv -o compare.png
```

`plot.py` берёт строки с `phase=matmul`, считает `GOP/s = 2·B·M·K·N / время` и строит график на каждую
группу случаев и пару типов. Линия — ядро; если файлов несколько, к имени ядра добавляется имя файла
(`adept: matmul`). `--time gpu` — GOP/s по `gpu_ms` вместо `wall_ms`, `--log` — логарифмическая шкала.
В stdout печатается GOP/s каждой точки.

## Что замеряется

Замер повторяет `adept::benchmark::measure` (`include/adept/benchmark/measure.hpp` в Adept):

1. `--warmup` холостых вызовов, затем ожидание конца работы на устройстве.
2. `--rounds` раундов. В каждом: ожидание, начало таймера GPU, `--iters` вызовов подряд, конец таймера GPU
   (он сам ждёт устройство) или ожидание.
3. На раунд — время на один вызов: время раунда / `--iters`. В отчёт идёт медиана по раундам.

| Колонка | Что это |
|---|---|
| `case` | имя случая; в `matmul_f16.csv` — с суффиксом `_f16_f32` или `_f16_f16` |
| `phase` | всегда `matmul`, как у GEMM в `op_bench` |
| `shape` | `BxMxKxN`, как у `op_bench`: размер пакета, M, K, N |
| `wall_ms` | полное время на вызов: от первого вызова блока до конца работы на устройстве |
| `gpu_ms` | время на GPU на вызов: от начала до конца блока по таймеру GPU, вместе с простоями между вызовами; `nan`, если ядро его не даёт |
| `cpu_ms` | время, пока CPU отправлял вызовы блока, на вызов |
| `kernel` | ядро |

- У CPU-ядер вызов синхронный: `wall_ms` ≈ `cpu_ms`, `gpu_ms` = `nan`.
- У `ggml_vulkan` и `cublas` вызов только ставит работу в очередь, ожидание одно на блок.
- `gpu_ms` есть только у ядер cuBLAS (CUDA events).
- Перед замером каждое ядро один раз считает задачу, результат сверяется с эталоном в double по 1024
  случайным элементам C. Ядро, не прошедшее проверку, не замеряется: в stderr — строка `FAILED`. NaN в
  результате — тоже ошибка.
- Данные и экземпляр ядра готовятся до замера (`prepare()`): загрузка на устройство, упаковка B не входят во
  время.

## Датасет случаев

Датасеты — CSV, одна строка — одна задача. `datasets/matmul.csv` — основной, f32; `datasets/matmul_f16.csv` —
синтетические формы в f16/f32 и f16/f16:

```
case,batch,m,k,n,dtype,acc,shared_b
square_2048,1,2048,2048,2048,f32,f32,0
square_2048_f16_f16,1,2048,2048,2048,f16,f16,0
batch_64x1024_shared_b,1024,64,64,64,f32,f32,1
```

| Колонка | Что это |
|---|---|
| `case` | имя случая, уникальное; часть до первого `_` — группа для `--case` |
| `batch`, `m`, `k`, `n` | пакет и размеры: A — m×k, B — k×n, C — m×n; в отчёте `shape` = `batch x m x k x n` |
| `dtype`, `acc` | тип входов и тип накопления и C: `f32`/`f32`, `f16`/`f32`, `f16`/`f16` |
| `shared_b` | `1` — одна B на весь пакет |

Файл проверяется при чтении: заголовок, 8 полей, положительные размеры, допустимые типы, уникальные имена.
Ошибка печатается с номером строки. Раскладка всегда `nn`. Ядро считает только строки, которые проходят его
`supports()`: `ggml_cpu` и `xnnpack` — f32, `ggml_vulkan` — f32, в режиме f16 — все пары, `cublas` и `cublas_tc` — все пары, `cublas_tf32` — f32.

Состав `matmul.csv` — 96 строк f32: 22 синтетические формы и 74 формы реальных моделей. В `matmul_f16.csv` —
только синтетические формы, по две строки f16 на форму (44).

Синтетические формы:

| Группа | Случаи | Что проверяет |
|---|---|---|
| `square` | `square_256`, `square_1024`, `square_2048`, `square_4096` | упор в вычисления, рост с размером |
| `inference` | `inference_m1`, `_m8`, `_m32`, `_m128`: M×4096×4096 | малое M, упор в память, как у инференса LLM |
| `inference` | `inference_k64`: 4096×64×4096 | малое K |
| `batch` | `batch_<N>x<B>`, N = 32, 64, 128, B = 1, 16, 1024 | пакеты малых матриц; в Adept пакет с N < 128 и B > 1 идёт одним dispatch (`gemm_t_c_batched`), с N = 128 — по dispatch на матрицу |
| `batch` | `batch_64x1024_shared_b` | одна B на весь пакет |
| `odd` | `odd_127`, `odd_1000` | размеры не кратны тайлам; 127 — сразу под порогом 128 выбора шейдера Adept |
| `matmul` | `matmul_attention`: пакет 8 по 100×32×100 | встроенный случай `op_bench` |

Формы реальных моделей (17.11). Параметры моделей взяты из их `config.json` на Hugging Face; как из них
получены формы, почему выбраны эти модели и чего не учитывают средние значения — в
[docs/shapes.md](docs/shapes.md). Обозначения: hidden — ширина модели, H и H_kv — головы запроса и K/V, d — размер
головы, I — ширина MLP, L — длина контекста.

| Группа | Случаи | Модель и слой | Источник |
|---|---|---|---|
| `da3s` | `da3s_qkv`, `_proj`, `_fc1`, `_fc2`: 1×1297×{384×1152, 384×384, 384×1536, 1536×384} | Depth Anything 3 small, линейные слои ViT; 1297 токенов при 504×504 | `apps/cpp/da3/da3_config.hpp` Adept, ветка `da3-impl` |
| `da3s` | `da3s_qk`, `da3s_pv`: 6×1297×64×1297, 6×1297×1297×64 | то же, QKᵀ и PV одного вида, 6 голов по 64 | там же |
| `da3s` | `da3s_qk_4v`, `da3s_pv_4v`: 6×5188×64×5188, 6×5188×5188×64 | глобальный блок внимания по 4 видам | `dino_backbone.cpp` Adept |
| `da3l` | `da3l_qkv`, `_fc1`, `_fc2`, `_qk` | DA3 large: hidden 1024, 16 голов, MLP 4096 | `depth-anything/DA3-LARGE` |
| `dec` | `dec_q8b_{qkv,up,down}_m{1,8,32}`: 1×M×{4096×6144, 4096×12288, 12288×4096} | Qwen3-8B, декодирование M токенов | `Qwen/Qwen3-8B` |
| `dec` | `dec_l8b_down_m{1,8,32}`: 1×M×14336×4096 | Llama-3.1-8B, down | `unsloth/Llama-3.1-8B-Instruct` (зеркало закрытого `meta-llama`) |
| `dec` | `dec_q06b_{qkv,o,up,down}_m{1,8,32}`: 1×M×{1024×4096, 2048×1024, 1024×3072, 3072×1024} | Qwen3-0.6B | `Qwen/Qwen3-0.6B` |
| `dec` | `dec_q8b_head`: 1×1×4096×151936 | Qwen3-8B, выход на словарь; B — 2.5 ГБ | `Qwen/Qwen3-8B` |
| `pre` | `pre_q8b_{qkv,up,down}`: M = 512; `pre_q8b_down_2048`: M = 2048 | Qwen3-8B, prefill пакетом `n_ubatch` 512 и `n_batch` 2048 | `Qwen/Qwen3-8B`, `common/common.h` llama.cpp |
| `moe` | `moe_q30b_{up,down}_{dec,pre}`: 8×1 и 128×32 по 2048×1536 и 768×2048 | Qwen3-30B-A3B, эксперты как пакет GEMM с разными B: 8 активных из 128 | `Qwen/Qwen3-30B-A3B` |
| `moe` | `moe_oss20b_{up,down}_dec`: 4×1×2880×{5760, 2880} | gpt-oss-20b, 4 активных эксперта | `openai/gpt-oss-20b` |
| `moe` | `moe_q36_up_dec`: 8×1×2048×1024; `moe_dsv4_up_dec`: 6×1×4096×4096 | Qwen3.6-35B-A3B, DeepSeek-V4-Flash | `config.json` моделей |
| `attn` | `attn_pre_d128_{qk,pv}`, `attn_pre_d128_qk_2048`: 32×L×128×L и 32×L×L×128 | Qwen3-8B, prefill 512 и 2048 | `Qwen/Qwen3-8B` |
| `attn` | `attn_pre_d256_qk`: 16×512×256×512 | Gemma 4, Qwen 3.6: d = 256 | `config.json` моделей |
| `attn` | `attn_dec_g4_{qk,pv}_{1024,4096,16384}`: 8×4×128×L и 8×4×L×128 | Qwen3-8B, Llama-3.1-8B: декодирование, 4 головы запроса на голову K/V в одном GEMM | там же |
| `attn` | `attn_dec_g8_qk_4096`, `attn_dec_oss_qk_win`, `attn_dec_gemma_qk_{win,glob}` | Qwen3-32B; gpt-oss, окно 128; Gemma-4-31B, окно 1024 и глобальный слой с d = 512 | `config.json` моделей |
| `attn` | `attn_dec_mqa_qk`: 1×64×512×512; `attn_dec_mla_{qk,pv}`: 1×128×576×2048, 1×128×2048×512 | DeepSeek-V4-Flash (одна голова K/V, top-k 512); DeepSeek-V3.2 (MLA, top-k 2048) | `config.json` моделей |
| `vit` | `vit_dinob_{qkv,fc1,qk}`, `vit_clipl_qkv`, `vit_siglip_{fc1,qk}` | DINOv2-base 518², CLIP ViT-L/14 224², SigLIP2 so400m/14 384² (d = 72) | `config.json` моделей |
| `asr` | `asr_whisper_{fc1,qk}`: 1×1500×1280×5120, 20×1500×64×1500 | Whisper-large-v3-turbo, энкодер | `openai/whisper-large-v3-turbo` |

Память в f32 (A + B + C): больше всех `dec_q8b_head` — 2.5 ГБ, почти всё — одна B; затем `moe_q30b_up_pre` —
1.7 ГБ. Все формы помещаются в 8 ГБ. Ограничение Vulkan на размер одного буфера на RTX 4070 не проверено.

Свой набор — копия файла со своими строками: `--matmul-cases my.csv` у обеих программ.

## Типы данных

Задача задаётся парой «тип входа / тип накопления», например `f16/f32`. Тип C совпадает с типом
накопления.

| Вход | Накопление и C | Допуск |
|---|---|---|
| f32 | f32 | `(K + 2)·2⁻²⁴` |
| f16 | f32 | `(K + 2)·2⁻²⁴` |
| f16 | f16 | `(K + 2)·2⁻¹¹` |

Если ядро само округляет входы (`cublas_tc` и `cublas_tf32` на f32, `ggml_vulkan_tc` и `ggml_vulkan_f16` на f32),
к допуску прибавляется `3·u`, где `u = 2⁻¹⁰`. Округление одного входа до f16 или TF32 даёт относительную
ошибку не больше `2⁻¹¹` при округлении к ближайшему и `2⁻¹⁰` при отбрасывании разрядов. Ошибка произведения
двух округлённых входов — меньше `3·u`. Ядро сообщает своё `u` через `GemmKernel::inputRounding()`.

- f16-входы генерируются сразу представимыми в f16, поэтому эталон в double остаётся точным.
- f16 хранится как `Half` из `half.hpp`; перевод float ↔ half свой, без `_Float16`.
- TF32 — только у `cublas_tf32`. i8 и транспонированных матриц нет: в Adept GEMM для i8 нет, случаи `op_bench` — `nn`.
- `ggml_cpu` и `xnnpack` — только f32; `ggml_vulkan` — f32, в режиме f16 — все пары; `cublas` и `cublas_tc` — все пары, `cublas_tf32` — f32.
- Основное сравнение — f32: только этот тип есть и в Adept, и у всех ориентиров. f16 — оценка сверху.

## Профилирование

Под профилировщиком гоняется один случай одного ядра, число вызовов задают `--warmup`, `--iters`, `--rounds`:

```bash
./build/gemm_bench --kernel ggml_vulkan --case-name square_2048 --warmup 5 --iters 20 --rounds 1
```

### NVIDIA

- Nsight Systems — таймлайн: отправки, dispatch-и на GPU, CUDA-ядра cuBLAS. Названия трассировок зависят от
  версии nsys, список — в `nsys profile --help`.

```bash
nsys profile --trace=vulkan,cuda,osrt -o gemm ./build/gemm_bench --kernel cublas --case-name square_2048
```

```bash
nsys stats gemm.nsys-rep
```

- Nsight Graphics — счётчики одного dispatch. GPU Trace с Capture Type «One-shot»: у bench нет кадров и
  present, обычный захват кадра не сработает.
- `cublas`: по именам CUDA-ядер в `nsys stats` видно, взял ли cuBLAS тензорные ядра (`cublas` против `cublas_tc`).
- `ggml_vulkan`: свои метки операций ggml ставит при `GGML_VK_DEBUG_MARKERS=1`, время каждой операции
  печатает при `GGML_VK_PERF_LOGGER=1`.

### M1 Pro (MoltenVK → Metal)

MoltenVK пишет захват Metal в файл, Xcode его открывает. Захват идёт всё время жизни `VkDevice`, поэтому
прогон должен быть коротким. Существующий файл MoltenVK не перезаписывает.

```bash
rm -rf ggml.gputrace
```

```bash
METAL_CAPTURE_ENABLED=1 MVK_CONFIG_AUTO_GPU_CAPTURE_SCOPE=1 MVK_CONFIG_AUTO_GPU_CAPTURE_OUTPUT_FILE=$PWD/ggml.gputrace ./build/gemm_bench --kernel ggml_vulkan --case-name square_1024 --warmup 1 --iters 2 --rounds 1
```

```bash
open ggml.gputrace
```

Открыть захват можно только в Xcode, Command Line Tools не хватит. Счётчиков в файле нет: их снимает Xcode, когда
выполняет записанные команды заново. Для этого на странице Summary в блоке Performance нажать Profile. Где искать
счётчики и стоимость строк шейдера — в [docs/profiling.md](docs/profiling.md).

Счётчики GPU без GUI записывает Instruments из командной строки (`xctrace`, входит в Xcode). `xcode-select`
переключать не нужно: Xcode задаётся через `DEVELOPER_DIR`.

```bash
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun xctrace record --no-prompt --template "Metal System Trace" --instrument "Metal GPU Counters" --output ggml.trace --launch -- $PWD/build/gemm_bench --kernel ggml_vulkan --case-name square_2048 --warmup 100 --iters 20 --rounds 5
```

```bash
python3 scripts/gpu_counters.py --state Maximum ggml.trace
```

`gpu_counters.py` усредняет счётчики по времени работы каждого compute pipeline. Кроме того, он печатает, в каком
состоянии производительности был GPU и какую долю времени ядра GPU выполнял работу других процессов. Pipeline
называются `main0 (<номер>)`. Перед записью нужно усыпить дисплей и дать прогрев не меньше 1 с (`--warmup`
побольше), иначе счётчики искажены. Условия записи, какие счётчики смотреть и как их проверяли — в
[docs/profiling.md](docs/profiling.md).

Пики устройств и замер пропускной способности памяти — в [docs/devices.md](docs/devices.md).

## Как добавить ядро

1. Создать `src/kernels/<name>.cpp`: наследники `GemmKernel` и `GemmInstance`
   и фабрика `std::unique_ptr<GemmKernel> make<Name>Kernel()`.
2. В `device()` вернуть устройство: у CPU-ядер `cpuName()` из `device.hpp`, у GPU-ядер имя GPU.
   В `backend()` — `cpu` у CPU-ядер (по нему работает `--device`). В `supports()` отсечь неподдерживаемые
   типы и формы.
3. В экземпляре: `run()` — один вызов; если он асинхронный, `sync()` ждёт конца всех поставленных вызовов, а
   `readC()` сначала вызывает `sync()`. Если ядро меряет время на GPU: в `beginGpuTiming()` поставить отметку
   начала, в `endGpuTimingMs()` — отметку конца, дождаться её и вернуть время блока в мс. Без них
   `endGpuTimingMs()` просто вызывает `sync()` и возвращает `nan`.
4. Добавить фабрику в `src/kernels/registry.cpp` и файл в `gemm_core` в `CMakeLists.txt`.
5. Ядро подключается опцией CMake `GEMM_WITH_<NAME>`: она добавляет файл ядра, библиотеку и define, а в
   `registry.cpp` фабрика стоит под `#ifdef GEMM_WITH_<NAME>`. Опцию добавить и в проверку `no kernels`.
   Пример — `xnnpack`.
