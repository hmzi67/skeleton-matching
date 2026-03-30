/**
 * auth.js — shared auth helpers used by every page.
 *
 * Token is stored in localStorage as "pm_token".
 * User info (username, role) stored as "pm_user" (JSON).
 */

const Auth = (() => {
  const TOKEN_KEY = 'pm_token';
  const USER_KEY  = 'pm_user';

  function getToken()  { return localStorage.getItem(TOKEN_KEY); }
  function getUser()   { const u = localStorage.getItem(USER_KEY); return u ? JSON.parse(u) : null; }
  function isLoggedIn(){ return !!getToken(); }
  function isAdmin()   { const u = getUser(); return u && u.role === 'admin'; }

  function save(token, user) {
    localStorage.setItem(TOKEN_KEY, token);
    localStorage.setItem(USER_KEY, JSON.stringify(user));
  }

  function logout() {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
    window.location.href = '/login';
  }

  /** Attach Authorization header to every fetch call */
  async function apiFetch(url, options = {}) {
    const token = getToken();
    const headers = { ...(options.headers || {}) };
    if (token) headers['Authorization'] = `Bearer ${token}`;
    // Don't set Content-Type for FormData — browser sets it with boundary
    if (!(options.body instanceof FormData)) {
      headers['Content-Type'] = headers['Content-Type'] || 'application/json';
    }
    const res = await fetch(url, { ...options, headers });
    if (res.status === 401) { logout(); }
    return res;
  }

  /**
   * Guard a page: redirect to /login if not logged in.
   * Pass role='admin' to also enforce admin-only.
   * Injects a nav bar into #navBar element if present.
   */
  function guard(role = null) {
    if (!isLoggedIn()) { window.location.replace('/login'); return; }
    if (role === 'admin' && !isAdmin()) { window.location.replace('/'); return; }
    _renderNav();
  }

  function _renderNav() {
    const el = document.getElementById('navBar');
    if (!el) return;
    const user = getUser();
    el.innerHTML = `
      <a class="nav-logo" href="/">Pose Matcher</a>
      <div class="nav-links">
        ${user.role === 'admin'
          ? `<a href="/admin">Dashboard</a>`
          : `<a href="/dashboard">My Uploads</a><a href="/upload">Upload Video</a>`
        }
        <span class="nav-user">👤 ${user.username}</span>
        <button class="nav-logout" onclick="Auth.logout()">Logout</button>
      </div>`;
  }

  return { getToken, getUser, isLoggedIn, isAdmin, save, logout, apiFetch, guard };
})();
