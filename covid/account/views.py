from django.shortcuts import render, redirect
from account.forms import *
from django.contrib.auth import logout, login, authenticate
from django.http import HttpResponse

# Create your views here.
def homepage_view(request):
    return render(request, 'account/home.html')


def login_view(request):

	context = {}

	user = request.user
	if user.is_authenticated: 
		return redirect("homepage")

	if request.POST:
		form = AccountAuthenticationForm(request.POST)
		if form.is_valid():
			email = request.POST['email']
			password = request.POST['password']
			user = authenticate(email=email, password=password)

			if user:
				login(request, user)
				return redirect("homepage")

	else:
		form = AccountAuthenticationForm()

	context['login_form'] = form

	# print(form)
	return render(request, "account/login.html", context)



def logout_view(request):
    logout(request)
    return redirect('homepage')

def register_view(request):
    valuenext= request.POST.get('next')
    if request.POST:
        form = RegistrationForm(request.POST)
        if form.is_valid():
            account = form.save()
            email = form.cleaned_data.get('email')
            password = form.cleaned_data.get('password')
            # account = authenticate(email=email, password=password)
            login(request, account)
            # Check urls.py we have given a name 'homepage' for the homepage view func
            return redirect('homepage')
            # if valuenext is None:
            #     return redirect('homepage')
            # else:
            #     return HttpResponse(valuenext)
        
        else:
            return render(request, 'account/register.html', context={'registration_form': form})
    
    else:
        form = RegistrationForm()
        return render(request, 'account/register.html', context={'registration_form': form})



def about_us_view(request):
    return render(request, 'account/about.html')

def contact_us_view(request):
    return render(request, 'account/contact.html')

def account_view(request):
    
    context = {}
    if not request.user.is_authenticated:
        return redirect('login')
    if request.POST:
        form = AccountUpdateForm(request.POST, instance=request.user)
        if form.is_valid():
            form.initial = {
                'email': request.POST['email'],
                'username': request.POST['username'],
            }
            form.save()
            context['success_message'] = 'Changes saved!!!'
    
    else:
        form = AccountUpdateForm(
            initial = {
                'email': request.user.email,
                'username': request.user.username
            }
        )
    context['account_form'] = form
    
    return render(request, 'account/account.html', context)


