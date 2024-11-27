from django.urls import path
from . import views

urlpatterns = [
    path('', views.welcome, name='welcome'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('book_list/', views.book_list, name='book_list'),
    path('books/<int:bookId>/', views.book_detail, name='book_detail'),  # Use bookId
    path('buy_book/<int:bookId>/', views.buy_book, name='buy_book'),  # Use pk
    path('search/', views.search_books, name='search_books'),
    path('logout/', views.logout_view, name='logout'),
    path('add_book/', views.add_book, name='add_book'),
    path('add_author/', views.add_author, name='add_author'),
    path('add_publisher/', views.add_publisher, name='add_publisher'),
    path('add_category/', views.add_category, name='add_category'),
    path('process_payment/<int:pk>/', views.process_payment, name='process_payment'),
]
