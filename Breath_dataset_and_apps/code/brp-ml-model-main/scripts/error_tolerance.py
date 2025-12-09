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