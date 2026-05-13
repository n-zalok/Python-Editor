export function saveToken(token) {
  localStorage.setItem("python-editor-token", token);
}

export function getToken() {
  return localStorage.getItem("python-editor-token");
}

export function clearToken() {
  localStorage.removeItem("python-editor-token");
}

export function isAuthenticated() {
  return Boolean(getToken());
}
