def acceptable_error(y_pred, y_true, i, EPSILON):
  matched = False
  batch_size = len(y_true)
  for j in range(1, EPSILON + 1):
    if i - j >= 0 and y_pred[i] == y_true[i - j]:
      matched = True
      break
    if i + j < batch_size and y_pred[i] == y_true[i + j]:
      matched = True
      break

  if not matched:
    return False  

  return True

def RR_error(cycle_true, cycle_pred, EPSILON):

  cycle_size = len(cycle_true)
  class_error = 0

  i = 0
  while i < cycle_size:
      if cycle_true[i] != cycle_pred[i]:
          class_error += 1
          error = cycle_pred[i]
          i += 1

          # pomijamy ciągły fragment tego samego błędu
          while i < cycle_size and (cycle_pred[i] != cycle_true[i] and cycle_pred[i] == error):
              i += 1
      else:
          i += 1

  if class_error  <= EPSILON:
    return True
  return False

