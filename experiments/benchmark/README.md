
Test intervened model performance on standard benchmarks:
* Aggregated: MMLU
* Math: (tiny)GSM8K
* Code: HumanEval


## Gemma2-2B; L10

tinyGSM8K:
* Prompt: 61.0 pm 14.8
  75, 70, 71, 29, 62, 61, 62, 70, 37, 73
* FSSV, Lang.: 10.7 pm 10.3
  15,  4,  2,  2, 19,  1,  6, 32,  3, 23
* ProSV, Lang.: 50.5 pm 6.7
  45, 55, 44, 52, 41, 45, 53, 65, 50, 55
* FSSV, SimPO: 5.6 pm 6.4
  10,  0,  4,  3,  6,  5,  1, 23,  1,  3
* ProSV, SimPO: 50.3 pm 4.6
  45, 52, 47, 55, 54, 49, 49, 60, 45, 47

tinyGSM8K (80% factor):
* FSSV, Lang.: 16.7 pm 11.3
  10, 11,  4,  7, 33, 10, 16, 38, 10, 28
* FSSV, SimPO: 10.7 pm 8.3
  10,  6,  3,  6, 13, 12,  6, 34,  9,  8

tinyGSM8K (50% factor):
* FSSV, Lang.: 33.6 pm 17.5
  18, 17, 14, 14, 51, 47, 43, 55, 20, 57
* FSSV, SimPO: 28.4 pm 11.3
  23, 22, 22, 10, 42, 36, 35, 49, 18, 27


## Gemma2-9B; L20

tinyMMLU:
* FSSV, Lang.: 55.4 pm 7.3
  72, 58, 58, 55, 49, 53, 48, 63, 49, 49
* ProSV, Lang.: 63.4 pm 11.6
  63, 68, 42, 64, 58, 72, 71, 77, 44, 75
* FSSV, SimPO: 47.2 pm 12.3
  19, 32, 44, 50, 46, 58, 58, 49, 58, 58
* ProSV, SimPO: 69.8 pm 11.2
  68, 78, 68, 68, 74, 63, 82, 76, 41, 80

tinyGSM8K:
* Prompt: 88.6 pm 5.8
  74, 92, 90, 92, 94, 90, 93, 91, 88, 82
* FSSV, Lang.: 8.6 pm 6.0
  17, 19, 10,  2,  7,  4, 15,  6,  2, 4
* ProSV, Lang.: 68.4 pm 17.2
  36, 42, 84, 72, 87, 80, 55, 78, 83, 67
* FSSV, SimPO: 4.2 pm 1.9
  2, 2, 5, 2, 2, 6, 7, 5, 6, 5
* ProSV, SimPO: 66.8 pm 20.3
  60, 17, 84, 64, 88, 80, 48, 73, 83, 71

tinyGSM8K (80% factor):
* FSSV, Lang.: 32.4 pm 14.9
  58, 50, 35, 17, 19, 30, 52, 18, 19, 26
* FSSV, SimPO: 20.8 pm 9.6
  10,  6, 22, 15, 21, 33, 39, 24, 14, 24

tinyGSM8K (50% factor):
* FSSV, Lang.: 75.2 pm 5.6
  83, 75, 79, 69, 75, 75, 85, 67, 75, 69
* FSSV, SimPO: 59.1 pm 18.3
  8, 48, 69, 67, 61, 71, 73, 69, 63, 62


## Qwen2.5-32B; L32

tinyMMLU:
* FSSV, Lang.: 41.4 pm 10.4
  51, 40, 43, 52, 35, 45, 45, 37, 15, 51
* PrOSV, Lang.: 68.5 pm 15.3
  64, 29, 76, 75, 79, 75, 79, 73, 54, 81
* FSSV, SimPO: 39.2 pm 10.4
  50, 35, 41, 52, 28, 34, 44, 38, 18, 52
* PrOSV, SimPO: 79.2 pm 7.1
  77, 62, 87, 84, 81, 81, 80, 74, 88, 78

tinyGSM8K:
* FSSV, Lang.: 6.6pm 3.3
  5, 6, 5, 2, 7, 11, 14, 5, 5, 6
* PrOSV, Lang.: 78.2 pm 12.7
  49, 85, 80, 75, 95, 83, 94, 77, 66, 78
* FSSV, SimPO: 6.9 pm 3.7
  12, 4, 5, 4, 2, 8, 13, 8, 10, 3
* PrOSV, SimPO: 79.2 pm 14.7
  96, 97, 84, 82, 88, 85, 34, 84, 78, 64

tinyGSM8K (80% factor):
* FSSV, Lang.: 32.7 pm 12.6
  51, 46, 27, 27, 26, 40, 52, 21, 21, 16
* FSSV, SimPO:
  27, 31, 24, 29, 15, 34, 48, 18, 15, 18

tinyGSM8K (50% factor):
* FSSV, Lang.: 82.0 pm 6.1
  89, 88, 83, 78, 85, 83, 90, 78, 70, 76
* FSSV, SimPO: 77.6 pm 7.4
  82, 87, 72, 82, 83, 79, 67, 63, 78, 83
