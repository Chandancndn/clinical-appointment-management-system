// Ask before a form that carries data-confirm is submitted (used for cancelling appointments).
document.addEventListener("submit", function (event) {
  var message = event.target.getAttribute("data-confirm");
  if (message && !window.confirm(message)) {
    event.preventDefault();
  }
});

// Label every table cell with its column heading, so the stylesheet can show rows as cards on a phone.
document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll("table").forEach(function (table) {
    var heads = Array.prototype.map.call(table.querySelectorAll("thead th"), function (th) { return th.textContent.trim(); });
    table.querySelectorAll("tbody tr").forEach(function (row) {
      var column = 0;
      Array.prototype.forEach.call(row.children, function (cell) {
        var span = cell.colSpan || 1;
        cell.setAttribute("data-label", span > 1 ? "" : (heads[column] || ""));
        column += span;
      });
    });
  });
});
