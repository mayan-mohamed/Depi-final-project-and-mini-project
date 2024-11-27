from django.shortcuts import render, get_object_or_404, redirect
from .models import Books
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import AuthenticationForm
from django.http import HttpResponse
import stripe
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from .models import Books, Author, Publisher, Category,BooksAuthor

from .forms import BookForm, AuthorForm, PublisherForm, CategoryForm, CustomUserCreationForm

stripe.api_key = settings.STRIPE_SECRET_KEY

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('book_list')
    else:
        form = AuthenticationForm()
    return render(request, 'app/login.html', {'form': form})

def register_view(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(data=request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = CustomUserCreationForm()
    return render(request, 'app/register.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('welcome')



def book_list(request):
    # Fetch all books and their associated authors
    books = Books.objects.all()
    # Use a separate query to fetch authors for each book
    book_author_map = {book.bookId: [ba.authorId for ba in BooksAuthor.objects.filter(bookId=book.bookId)] for book in books}
    return render(request, 'app/book_list.html', {'books': books, 'book_author_map': book_author_map})


from django.http import HttpResponse


def book_detail(request, bookId):
    # Fetch the book by its ID
    book = get_object_or_404(Books, bookId=bookId)
    
    # Fetch related details: publisher, category, and authors
    publisher = book.publisher
    categories = book.bookscategories_set.all()
    authors = book.booksauthor_set.all()

    context = {
        'book': book,
        'publisher': publisher,
        'categories': categories,
        'authors': authors,
    }
    
    return render(request, 'app/book_details.html', context)




def buy_book(request, bookId):
    print(f"bookId received: {bookId}")
    book = get_object_or_404(Books,bookId=bookId)
    price_in_cents = int(book.price * 100)
    return render(request, 'app/buy_book.html', {
        'book': book,
        'stripe_publishable_key': settings.STRIPE_PUBLISHABLE_KEY,
        'price_in_cents': price_in_cents,

    })

@csrf_exempt
def process_payment(request, bookId):
    book = get_object_or_404(Books, bookId=bookId)  # Updated model name
    token = request.POST.get('stripeToken')

    try:
        charge = stripe.Charge.create(
            amount=int(book.price * 100),  # amount in cents
            currency='usd',
            description=f'Purchase of {book.title}',
            source=token
        )
        return HttpResponse('Payment Successful')
    except stripe.error.StripeError as e:
        return HttpResponse('Payment Failed: ' + str(e))

def search_books(request):
    query = request.GET.get('q')
    if query:
        books = Books.objects.filter(title__icontains=query)  # Updated model name
    else:
        books = Books.objects.all()  # Updated model name
    return render(request, 'app/book_list.html', {'books': books})

def add_book(request):
    if request.method == 'POST':
        form = BookForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('book_list')
    else:
        form = BookForm()
    return render(request, 'app/add_book.html', {'form': form})

def add_author(request):
    if request.method == 'POST':
        form = AuthorForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('book_list')
    else:
        form = AuthorForm()
    return render(request, 'app/add_author.html', {'form': form})

def add_publisher(request):
    if request.method == 'POST':
        form = PublisherForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('book_list')
    else:
        form = PublisherForm()
    return render(request, 'app/add_publisher.html', {'form': form})

def add_category(request):
    if request.method == 'POST':
        form = CategoryForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('book_list')
    else:
        form = CategoryForm()
    return render(request, 'app/add_category.html', {'form': form})

def welcome(request):
    return render(request, 'app/welcome.html')